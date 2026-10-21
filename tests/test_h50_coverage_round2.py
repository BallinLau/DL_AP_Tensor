"""Round-two non-optimizing contracts; synthetic fixtures are not economics."""
from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace
import json,math
import pytest
import torch
import pandas as pd
from training.inner_fixed_point import InnerFPCollocationBank
from data.finite_direct_uniform_firm_macro import FirmMacroCoverageConfig,configured_macro_bank,uniform_macro_bank
from data.finite_direct_runtime import conditional_child_bank,firm_bp_parent_rows
from analysis.economic_config import AnalysisEconomicConfig
from analysis.finite_direct_continuation_state import state_digest
from data.finite_direct_joint_coverage import JointCoverageConfig,JointReservoir,joint_bin,bin_direction,build_joint_bank,eligible_templates
from data.finite_direct_long_horizon_runtime import certify_template_accounting
from data.finite_direct_long_horizon_empirical import committed_templates
from data.finite_direct_sampling import file_hash
from analysis.nested_equilibrium_utils import atomic_torch_save,atomic_write_json
from analysis.h50_effective_coverage import lineage_statistics


@pytest.fixture(autouse=True)
def no_optimization(monkeypatch):
    def stop(*a,**k):raise AssertionError('optimization forbidden')
    for c in (torch.optim.Adam,torch.optim.AdamW,torch.optim.LBFGS):monkeypatch.setattr(c,'step',stop)
    monkeypatch.setattr(torch.Tensor,'backward',stop)


def donor(n=128):
    g=torch.Generator().manual_seed(123);s=torch.rand(n,7,generator=g);s[:,1]=torch.randn(n,generator=g);s[:,2]=torch.arange(n)%2
    return InnerFPCollocationBank(s,torch.arange(n)%2,1,'train',dict(true_macro_context=s[:,5:7]+1,
        source_kind=['ON_POLICY']*(n//2)+['CALCULATED_MACRO_COLLOCATION']*(n-n//2),path=torch.arange(n)))


def test_uniform_input_no_cal_label_micro_preserved_and_both_eta():
    d=donor();old=state_digest(d.payload());u=uniform_macro_bank(d,seed=4)
    assert torch.equal(u.parent_states[:,:5],d.parent_states[:,:5])
    assert set(u.parent_states[:,2].tolist())=={0,1}
    assert u.parent_states[:,5].min()>=-6 and u.parent_states[:,5].max()<=2
    assert u.parent_states[:,6].min()>=-2 and u.parent_states[:,6].max()<=5
    assert 'true_macro_context' not in u.metadata and u.fingerprint!=d.fingerprint
    assert state_digest(d.payload())==old


def test_explicit_hybrid_composition_and_default_off():
    d=donor();assert configured_macro_bank(d,FirmMacroCoverageConfig(),seed=8) is d
    with pytest.raises(ValueError,match='explicit'):configured_macro_bank(d,FirmMacroCoverageConfig(firm_macro_coverage_enabled=True),seed=8)
    c=FirmMacroCoverageConfig(firm_macro_coverage_enabled=True,firm_macro_coverage_fraction=.25)
    u=configured_macro_bank(d,c,seed=8)
    assert u.metadata['source_composition']=={'FIRM_ON_POLICY':48,'FIRM_CALCULATED_MACRO_COLLOCATION':48,'FIRM_UNIFORM_MACRO_COLLOCATION':32}
    assert u.fingerprint==configured_macro_bank(d,c,seed=8).fingerprint


def test_conditional_children_m_bp_rebuilt_at_new_context():
    class Macro(torch.nn.Module):
        def forward_step(self,*,x_prev,x_curr,hatcf_prev,lnkf_prev,return_physical):
            shape=x_curr.shape;one=torch.ones_like(x_curr)
            return one[:,0],one,(.9+.01*hatcf_prev[:,None]).expand(shape),hatcf_prev[:,None].expand(shape),lnkf_prev[:,None].expand(shape)
    m=Macro();before=state_digest(m.state_dict());d=donor();u=uniform_macro_bank(d,seed=3)
    kw=dict(seed=7,n_child_shocks=2,economic=AnalysisEconomicConfig.from_current_config(),device='cpu')
    a=conditional_child_bank(m,d,**kw);b=conditional_child_bank(m,u,**kw)
    torch.testing.assert_close(b.children[:,:,5:7],u.parent_states[:,None,5:7].expand(-1,2,-1))
    assert not torch.equal(a.children,b.children) and not torch.equal(a.metadata['raw_m'],b.metadata['raw_m'])
    torch.testing.assert_close(firm_bp_parent_rows(u)[:,7:9],u.parent_states[:,5:7])
    assert b.metadata['parent_fingerprint']==u.fingerprint and state_digest(m.state_dict())==before


def test_incremental_occupancy_and_fourteen_bin_capacity():
    cfg=JointCoverageConfig();r=JointReservoir(8192,cfg)
    for j in range(8192):r.add(dict(parent_id=j,coverage_bin=(j%14//8,j%14%8),coverage_family=cfg.families[j%7]))
    assert r.count==14*256 and sum(r.eligible_count_by_bin.values())==8192
    assert sum(r.occupancy_by_bin.values())==r.count==sum(r.occupancy_by_bin_family.values())
    for (cell,family),values in r.buckets.items():assert r.occupancy_by_bin_family[cell,family]==len(values)


@pytest.mark.parametrize('h,k,expected',[(-3,3,'H_LOW/ELL_IN'),(1,3,'H_HIGH/ELL_IN'),(-1,-3,'H_IN/ELL_LOW'),(-1,7,'H_IN/ELL_HIGH'),(-3,-3,'H_LOW/ELL_LOW')])
def test_directional_overflow(h,k,expected):
    c=JointCoverageConfig();assert bin_direction(joint_bin(h,k,c),c)==expected
    assert joint_bin(float('nan'),k,c) is None


def test_bounded_refill_only_distinct_certified_candidates(tmp_path,monkeypatch):
    from test_h50_joint_coverage import reference
    import data.finite_direct_joint_coverage as m
    import analysis.run_fc1_self_consistent_macro as b
    monkeypatch.setattr(b,'hashes',lambda x:{'firm':'same'})
    def certify(sim,s,**kw):
        rows=[dict(parent_candidate_id=int(pid),C_raw_t1=1.,K_cal_t1=10.,Hatc_cal_t1=-1.,LnK_cal_t1=3.,economic_state_valid=True) for pid in s['transition_path_id']]
        return s,rows,list(range(len(rows))),[{} for _ in rows]
    monkeypatch.setattr(m,'calibrate_initial',certify)
    kw=dict(reference=reference(4),templates=[],outer=12,target=16,cfg=replace(JointCoverageConfig(),max_candidates=20,bin_cap=1,candidate_batch_size=4),output=tmp_path,
        simulator_factory=lambda n:SimpleNamespace(models={},device='cpu'),economic={})
    bank,plan,stats=build_joint_bank(**kw);audit=stats['capacity_audit']
    assert len(plan['parents'])==len({p['parent_id'] for p in plan['parents']})==16
    assert audit['reachable_capacity_under_current_caps']==1 and audit['capacity_refill_count']==15
    assert audit['capacity_shortfall']==15 and audit['economic_shortfall']==0
    assert build_joint_bank(**kw)[0]['fingerprint']==bank['fingerprint']


def test_empirical_next_outer_sha_stage_and_same_node_evidence(tmp_path):
    t=dict(split='train',source_outer=12,source_time=3,source_branch=0,source_path=10,cluster_id='p10',template_id='p10:t3',
        b_sorted=[.1,.2],b_by_capital_quartile=[[.1]]*4,post_decision_accounting_valid='NOT_YET_EVALUATED')
    assert not eligible_templates([t],outer=13,cfg=JointCoverageConfig(),committed=True)[0]
    items=[dict(dynamic_template=t,split='train')]
    certify_template_accounting(items,[dict(parent_candidate_id=10,economic_state_valid=True,C_raw_t1=1.,K_cal_t1=2.)])
    shard=tmp_path/'one.pt';atomic_torch_save(shard,dict(templates=items))
    atomic_write_json(tmp_path/'manifest.json',dict(provenance=dict(stage='POST_SDF_TRAIN_EMPIRICAL_SOURCE',split='train',outer=12),shards={'one.pt':file_hash(shard)}))
    source=dict(path=str(tmp_path),outer=12,manifest_sha256=file_hash(tmp_path/'manifest.json'))
    with pytest.raises(ValueError,match='activation'):list(committed_templates([source],outer=12))
    loaded=list(committed_templates([source],outer=13));eligible,_=eligible_templates(loaded,outer=13,cfg=JointCoverageConfig(),committed=True)
    assert len(eligible)==1 and eligible[0]['source_time']==3 and eligible[0]['accounting_evidence']['parent_id']==10


def test_ess_counts_lineages_not_child_independence():
    s=lineage_statistics([100,97]);assert s['ESS']==pytest.approx(197**2/(100**2+97**2))
    assert s['largest_lineage_share']==100/197 and s['unique_lineages']==2


def test_matched_evaluator_requires_same_macro_and_distinct_firm_input_grid():
    from analysis.check_h50_round2_matched import matched_payloads,static_grid
    m=dict(policy_value={'w':torch.tensor([1.])},sdf_fc1={f'{prefix}.w':torch.tensor([1.]) for prefix in ('sdf_model','value_model','fc1_model.FC1_C','fc1_model.FC1_K')})
    a={'transaction':{'models':m}};b=deepcopy(a);b['transaction']['models']['policy_value']['w']+=1
    ha,hb=matched_payloads(a,b);assert ha['firm']!=hb['firm']
    b['transaction']['models']['sdf_fc1']['sdf_model.w']+=1
    with pytest.raises(ValueError,match='UNMATCHED'):matched_payloads(a,b)
    g=static_grid();p=g.parent_states
    assert torch.equal(p[p[:,2]==0][:,[0,1,3,4,5,6]],p[p[:,2]==1][:,[0,1,3,4,5,6]])


def test_invalid_terminal_suffix_is_counted_without_fake_labels(tmp_path):
    from analysis.h50_effective_coverage import collect_effective_coverage
    class Store:
        def rows(self):yield pd.DataFrame([dict(split='train',parent_candidate_id=1,t=10,child_branch=b,economic_state_valid=False) for b in (0,1)])
    f=collect_effective_coverage(Store(),tmp_path,JointCoverageConfig())
    assert f.raw_rows.sum()==0 and f.complete_sdf_pairs.sum()==0


def test_regret_interface_and_control_budget_identical(monkeypatch):
    import inspect
    from training.bp_grid_teacher import BPGridTeacher
    assert 'bp_pred' in inspect.signature(BPGridTeacher.compute).parameters
    import analysis.run_uniform_macro_frozen_firm as r
    a=r.plan();monkeypatch.setattr(r,'DISTRIBUTION_MODE','ORIGINAL');b=r.plan()
    for key in ('firm_batch_size','firm_inner_max','p_adam_steps','p_lbfgs_max_iter','q_fit_steps','rng_seed','optimizer_semantics'):assert a[key]==b[key]
    assert a['effective_uniform_fraction']==1 and b['effective_uniform_fraction']==0
