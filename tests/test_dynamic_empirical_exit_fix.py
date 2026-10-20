"""CPU production-transition forward and fake-journal tests. NO optimizers."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import inspect
import json
import numpy as np
import pytest
import torch
from analysis.audit_dynamic_empirical_timing import fixture
from data.finite_direct_runtime import expand_after_parent_exit,cache_provenance,verify_cache,generate_dataset,OPERATOR
from data.simulate_ts_parallel import _expand_branches_batched,_apply_entry_batched,_process_node_batched
from data.entry import compute_entry_reference,capital_growth_decomposition
from data.finite_direct_dynamic_empirical import (template,initial_pool,next_pool,resample_train,capture_templates,
    save_bundle,load_bundle,CONFIG,fingerprint,checked)
from analysis.finite_direct_continuation_state import state_digest,safe_load
from analysis.finite_direct_outer4_state import StageJournal
from analysis.finite_direct_dynamic_approval import approval_binding,require_approval,prepare_initial_pool
from analysis.run_finite_direct_large_continuation import stages,training_plan,ContinuationBackend


def synthetic(outer=4,count=8,firms=16):
    n=2*count;f=firms
    state=dict(b=torch.full((n,f),.25),bp=torch.full((n,f),.25),K=torch.arange(1,f+1).float().repeat(n,1),
        alive=torch.ones(n,f,dtype=torch.bool),transition_path_id=torch.arange(n),firm_id=torch.arange(f).repeat(n,1),z=torch.zeros(n,f),eta=torch.zeros(n,f),i=torch.ones(n,f))
    schedule=dict(split='train',seed=3,parents=[dict(parent_id=j,sampling_source='BETA' if j<count else 'EMPIRICAL',
        child_shock_keys=[dict(branch=0,key=j),dict(branch=1,key=j)],empirical_template_id='old') for j in range(n)])
    schedule['fingerprint']=fingerprint(schedule)
    bank=dict(state=state,fingerprint=state_digest(state),schedule_fingerprint=schedule['fingerprint'])
    original=[]
    for j in range(4):
        original.append(template(np.linspace(.1,.5,f),np.arange(1,f+1),identity=f'old{j}',source=dict(split='train',source_path=j,source_time=0)))
    return bank,schedule,initial_pool(dict(templates=original))


def source_captures(outer=4):
    bank,_,_=synthetic();s=bank['state'];n=len(s['b'])
    s.update(transition_previous_alive=s['alive'].clone())
    return capture_templates(s,split='train',outer=outer,branch=0,model_hashes={'firm':f'f{outer}'}) + capture_templates(s,split='train',outer=outer,branch=1,model_hashes={'firm':f'f{outer}'})


def test_both_children_remove_only_parent_defaults_and_keep_ledger_inputs():
    sim,p=fixture();p['bar_i']=torch.tensor([[1.,.25,0.]])
    before=state_digest(p);raw=_expand_branches_batched(sim,deepcopy(p));fixed=expand_after_parent_exit(sim,deepcopy(p))
    for a,b in zip(raw,fixed):
        assert b['alive'].tolist()==[[False,True,False]]
        for name in ('b','K','z','eta','i','transition_previous_alive','transition_previous_K','event_reference_K','event_reference_N'):
            torch.testing.assert_close(a[name],b[name])
        assert b['b'][0,1]==p['b'][0,1]
        torch.testing.assert_close(b['K'][0,1],p['K'][0,1]*(1+.25*(sim.g-1)))
        assert b['firm_mass'][0,0]==0
    assert state_digest(p)==before


def test_exit_disabled_is_exact_original_transition():
    sim,p=fixture();sim.enable_exit=False
    assert state_digest(expand_after_parent_exit(sim,deepcopy(p)))==state_digest(_expand_branches_batched(sim,deepcopy(p)))


def test_entry_reference_is_original_pre_transition_population_not_survivor_scale():
    sim,p=fixture();c=expand_after_parent_exit(sim,p)[0]
    args=dict(entry_capital_ratio=.1,entry_size_ratio=1.)
    old=compute_entry_reference(c['transition_previous_K'],c['transition_previous_alive'],**args)
    post=compute_entry_reference(c['K'],c['alive'],**args)
    assert old.reference_N.item()==2 and post.reference_N.item()==1
    torch.testing.assert_close(c['event_reference_K'],old.reference_K)
    d=capital_growth_decomposition(p['firm_id'][p['alive']],p['K'][p['alive']],c['firm_id'][c['alive']],c['K'][c['alive']])
    assert d['K_exit_old'].item()==1 and d['capital_accounting_residual'].item()==0


def test_actual_parent_policy_default_ledger_then_both_children_exit():
    from test_entry_value_cost import _value_cost_sim,_SimulationPolicy
    from data.simulate_ts_parallel import _initialize_batched_state
    sim,_=_value_cost_sim(bp_source='grid',stable_identity_rng=True);sim.branch_num=2
    sim.models['policy_value']=_SimulationPolicy(default_zero_debt=True)
    parent=_initialize_batched_state(sim,4);parent['b'][0,:2]=torch.tensor([0.,.5])
    _process_node_batched(sim,parent,0,-1)
    assert parent['bar_z'][0,0]==1 and parent['bar_z'][0,1]==0
    ledger={k:v.clone() for k,v in parent.items() if k.startswith('accounting_') and torch.is_tensor(v)}
    for child in expand_after_parent_exit(sim,parent):
        assert child['alive'][0,:2].tolist()==[False,True]
        torch.testing.assert_close(child['transition_previous_alive'],parent['alive'])
        torch.testing.assert_close(child['transition_previous_K'],parent['K'])
        assert child['b'][0,1]==(.6 if parent['eta'][0,1]==1 else parent['b'][0,1])
    for k,v in ledger.items():torch.testing.assert_close(parent[k],v,equal_nan=True)


def test_real_production_entry_and_child_policy_preserve_birth_timing():
    from test_entry_value_cost import _value_cost_sim
    sim,_=_value_cost_sim();sim.enable_exit=True
    from data.simulate_ts_parallel import _initialize_batched_state
    p=_initialize_batched_state(sim,4)
    p['bar_z']=torch.zeros_like(p['b']);p['bar_z'][0,0]=1
    p['bar_i']=torch.zeros_like(p['b']);p['bp']=p['b'].clone()
    p['hatc_cal']=torch.zeros(1);p['lnk_cal']=torch.zeros(1)
    child=expand_after_parent_exit(sim,p)[0]
    assert not child['alive'][0,0]
    oldid=p['firm_id'][0,0].item()
    child=_apply_entry_batched(sim,child,20)
    captured=capture_templates(child,split='train',outer=4,branch=0,model_hashes={})
    assert captured[0]['dynamic_template']['firm_count']==int(child['alive'].sum())
    assert oldid not in child['firm_id'][child['alive']].tolist()
    assert (child['b'][child['entry']>.5]==0).all()
    _process_node_batched(sim,child,1,0)
    assert not child['alive'][0,0]  # policy cannot revive previous default


def test_investment_compare_birth_state_and_current_default_not_used_to_select_templates():
    from test_entry_investment_compare import _investment_sim,_PositiveDeltaConfig
    from data.simulate_ts_parallel import _initialize_batched_state
    sim=_investment_sim(horizon=1,config=_PositiveDeltaConfig);sim.branch_num=2
    p=_initialize_batched_state(sim,2);p['hatc_cal']=torch.zeros(1);p['lnk_cal']=torch.zeros(1)
    p['bar_z']=torch.zeros_like(p['b']);p['bar_i']=torch.zeros_like(p['b']);p['bp']=p['b'].clone()
    for branch,c in enumerate(expand_after_parent_exit(sim,p)):
        c=_apply_entry_batched(sim,c,20);entry=c['entry']>.5
        assert entry.any() and (c['b'][entry]==0).all() and (c['eta'][entry]==0).all()
        torch.testing.assert_close(c['i'][entry],c['entry_cost'][entry])
        torch.testing.assert_close(c['K'][entry],c['K_birth'][entry])
        snap=capture_templates(c,split='train',outer=4,branch=branch,model_hashes={})
        pre=deepcopy(snap)
        _process_node_batched(sim,c,1,branch)
        assert snap==pre and (c['bar_i'][entry]==0).all()


def test_production_adapter_captures_predecision_before_current_policy(tmp_path,monkeypatch):
    import data.finite_direct_runtime as runtime
    from test_entry_value_cost import _value_cost_sim,_SimulationPolicy
    from data.simulate_ts_parallel import _initialize_batched_state
    from config import HyperParams
    sim,_=_value_cost_sim(bp_source='grid',stable_identity_rng=True);sim.branch_num=2
    sim.models['policy_value']=_SimulationPolicy(default_zero_debt=True)
    parent=_initialize_batched_state(sim,4);parent['b'][0,:2]=torch.tensor([0.,.5])
    monkeypatch.setattr(runtime,'simulator',lambda *a,**kw:sim)
    monkeypatch.setattr(runtime,'hashes',lambda models:{'firm':'f'})
    bank=dict(state=parent,fingerprint='initial');schedule=dict(split='train',seed=1,parents=[dict(parent_id=0,child_shock_keys=[0,1])])
    episode=SimpleNamespace(device=torch.device('cpu'),hyperparams=HyperParams())
    frame,captures,manifest=runtime.generate_dataset({},episode,bank,schedule,{'fingerprint':'s'},tmp_path,
        outer=4,stage=CONFIG['source'],capture=False,predecision_templates=True)
    assert len(frame)==2 and len(captures)==2
    for capture in captures:
        t=capture['dynamic_template'];assert t['parent_default_count']==1
        # New debt-zero entrants are included BEFORE their own default decision.
        assert 0. in t['b_sorted'] and t['post_decision_accounting_valid']=='NOT_YET_EVALUATED'


def test_outer4_is_locked_t0_and_next_outers_use_previous_sources():
    bank,schedule,p=synthetic();b,s,h=resample_train(bank,schedule,p,outer=4,empirical_count=8)
    assert state_digest(b['state'])==state_digest(bank['state']) and h==[]
    p5=next_pool(p,source_captures(4),source_outer=4,source_hashes={'firm':'f4'},source_manifest_sha='m4')
    b5,s5,h5=resample_train(bank,schedule,p5,outer=5,empirical_count=8)
    assert {h['source_outer'] for h in h5 if h['pool_source']=='RECENT'}=={4}
    p6=next_pool(p5,source_captures(5),source_outer=5,source_hashes={'firm':'f5'},source_manifest_sha='m5')
    _,_,h6=resample_train(bank,schedule,p6,outer=6,empirical_count=8)
    assert {h['source_outer'] for h in h6 if h['pool_source']=='RECENT'}=={5}
    assert p6['recent']!=p['archive']


def test_beta_capital_other_shocks_ids_and_crn_unchanged():
    bank,schedule,p=synthetic();p=next_pool(p,source_captures(),source_outer=4,source_hashes={'firm':'f4'},source_manifest_sha='m')
    old=state_digest(bank);b,s,h=resample_train(bank,schedule,p,outer=5,empirical_count=8)
    torch.testing.assert_close(b['state']['b'][:8],bank['state']['b'][:8])
    for name in ('K','alive','z','eta','i','firm_id','transition_path_id'):
        torch.testing.assert_close(b['state'][name],bank['state'][name])
    assert all(a['child_shock_keys']==c['child_shock_keys'] for a,c in zip(s['parents'],schedule['parents']))
    assert state_digest(bank)==old
    assert len([r for r in s['parents'] if r['sampling_source']=='BETA'])==8


def test_exact_4096_recent_archive_counts_and_cdf_reproducibility():
    bank,schedule,p=synthetic(count=4096);p=next_pool(p,source_captures(),source_outer=4,source_hashes={'firm':'f4'},source_manifest_sha='m')
    b,s,h=resample_train(bank,schedule,p,outer=5)
    assert sum(r['pool_source']=='RECENT' for r in h)==2867
    assert sum(r['pool_source']=='ARCHIVE' for r in h)==1229
    b2,s2,h2=resample_train(bank,schedule,p,outer=5)
    assert state_digest(b)==state_digest(b2) and s==s2 and h==h2
    assert set(b['state']['b'][4096:].flatten().tolist())<=set(np.array(p['recent'][0]['b_sorted']+p['archive'][0]['b_sorted'],dtype=np.float32).tolist())


@pytest.mark.parametrize('field,value',[('split','validation'),('source_outer',5),('source_stage','D2'),('source_time',0),('source_kind','POST_DECISION_ACCOUNTING_VALID')])
def test_reject_validation_future_and_post_decision_sources(field,value):
    _,_,p=synthetic();c=source_captures();c[0]['dynamic_template'][field]=value
    if field=='split':c[0]['split']=value
    with pytest.raises(ValueError):next_pool(p,c,source_outer=4,source_hashes={'firm':'f4'},source_manifest_sha='m')


def test_capture_does_not_filter_child_future_failure_or_current_default():
    bank,_,_=synthetic();s=bank['state'];s['bar_z']=torch.ones_like(s['b'])
    s['transition_previous_alive']=s['alive'].clone()
    c=capture_templates(s,split='train',outer=4,branch=0,model_hashes={})
    assert len(c)==16 and all(r['dynamic_template']['firm_count']==16 for r in c)
    assert all(r['dynamic_template']['post_decision_accounting_valid']=='NOT_YET_EVALUATED' for r in c)
    with pytest.raises(ValueError):capture_templates(s,split='validation',outer=4,branch=0,model_hashes={})


def test_empty_recent_is_failure_not_renamed_archive():
    _,_,p=synthetic()
    with pytest.raises(ValueError,match='NO_RECENT'):next_pool(p,[],source_outer=4,source_hashes={},source_manifest_sha='m')


def test_archive_clusters_same_lineage_across_outers_instead_of_counting_snapshots():
    _,_,p=synthetic()
    for outer in (4,5,6):
        p=next_pool(p,source_captures(outer),source_outer=outer,source_hashes={'firm':f'f{outer}'},source_manifest_sha=f'm{outer}')
    dynamic=[t for t in p['archive'] if t['source_outer']>=4]
    assert len(dynamic)==64 and len({t['cluster_id'] for t in dynamic})==16


def test_cluster_not_sibling_sampling_and_conditional_fallback():
    bank,schedule,p=synthetic(count=1000);captures=source_captures()
    # A Parent with one usable pre-decision branch has the same cluster weight
    # as one with two, rather than half the probability.
    captures=captures[:-1]
    p=next_pool(p,captures,source_outer=4,source_hashes={'firm':'f4'},source_manifest_sha='m')
    _,_,h=resample_train(bank,schedule,p,outer=5,empirical_count=1000)
    recent=[r for r in h if r['pool_source']=='RECENT']
    counts=pd_counts([r['cluster_id'] for r in recent])
    assert max(counts.values())<90 and len(counts)==16
    assert all(r['fallback_fraction']==1 for r in h)  # four points/quartile <8


def test_nonfallback_conditional_cdf_keeps_capital_quartile_support():
    bank,schedule,p=synthetic(firms=64);s=deepcopy(bank['state'])
    s['b'][:]=torch.tensor([.1]*16+[.3]*16+[.6]*16+[.9]*16)
    s['transition_previous_alive']=s['alive'].clone()
    captures=capture_templates(s,split='train',outer=4,branch=0,model_hashes={'firm':'f4'})
    pool=next_pool(p,captures,source_outer=4,source_hashes={'firm':'f4'},source_manifest_sha='m')
    b,_,h=resample_train(bank,schedule,pool,outer=5,empirical_count=8)
    for row in h:
        if row['pool_source']=='RECENT':
            assert row['fallback_fraction']==0
            torch.testing.assert_close(b['state']['b'][row['parent_id']],s['b'][row['parent_id']])


def pd_counts(values):
    from collections import Counter
    return Counter(values)


def test_content_sha_resume_no_resample_and_tamper_rejection(tmp_path):
    bank,schedule,p=synthetic();b,s,h=resample_train(bank,schedule,p,outer=4,empirical_count=8)
    record=save_bundle(tmp_path,p,b,s,h);a=load_bundle(record);second=save_bundle(tmp_path,p,b,s,h)
    assert record==second and state_digest(a)==state_digest(load_bundle(record))
    target=Path(record['bank']['path']);target.write_bytes(target.read_bytes()+b'x')
    with pytest.raises(ValueError,match='SHA'):load_bundle(record)


def test_journal_commits_pool_and_restores_references_without_reexecution(tmp_path):
    from test_finite_direct_outer4_runner import FakeBackend
    b=FakeBackend();j=StageJournal(tmp_path,dict(operator=OPERATOR),b)
    j.run('outer_004/EMPIRICAL_POOL_COMMIT',lambda:b.execute(4,'EMPIRICAL_POOL_COMMIT',{}))
    restored=FakeBackend();j2=StageJournal(tmp_path,dict(operator=OPERATOR),restored,resume=True)
    j2.run('outer_004/EMPIRICAL_POOL_COMMIT',lambda:pytest.fail('must not resample committed pool'))
    assert restored.state==b.state


def test_real_pool_commit_and_activation_recover_identical_bank(tmp_path):
    from data.finite_direct_sampling import file_hash
    from analysis.nested_equilibrium_utils import atomic_torch_save,atomic_write_json,stable_json_hash
    bank,schedule,p=synthetic(count=4096)
    b4,s4,h4=resample_train(bank,schedule,p,outer=4)
    initial=save_bundle(tmp_path/'initial',p,b4,s4,h4)
    source=tmp_path/'source';source.mkdir();shard=source/'batch.pt'
    atomic_torch_save(shard,dict(captures=source_captures()))
    atomic_write_json(source/'manifest.json',dict(shards={'batch.pt':file_hash(shard)}))
    validation_file=tmp_path/'validation_initial_bank.pt';validation_file.write_bytes(b'fixed validation')
    class StateOnlyBackend(ContinuationBackend):
        def __init__(self):
            self.root=tmp_path/'run';self._initial_train=bank;self._initial_schedule=schedule
            self.args=SimpleNamespace(sampling_dir=tmp_path)
            self.banks={'validation':{'fixed':1}};self.schedules={'validation':{'CRN':'fixed'}}
            self._validation_identity=(file_hash(validation_file),stable_json_hash(self.schedules['validation']))
            self.state=dict(dynamic_banks={'4':initial},active_dynamic_outer=4,dynamic_sources={'4':dict(
                path=str(source),manifest_sha256=file_hash(source/'manifest.json'),model_hashes={'firm':'f4'})})
            self._activate_dynamic(4)
        def snapshot(self):return deepcopy(self.state)
        def restore(self,value):
            self.state=deepcopy(value);self._activate_dynamic(self.state['active_dynamic_outer'])
    b=StateOnlyBackend();j=StageJournal(tmp_path/'journal',{'operator':OPERATOR},b)
    old_bank=state_digest(b.banks['train'])
    result=j.run('outer_004/EMPIRICAL_POOL_COMMIT',lambda:b._pool_commit(4))
    assert result['optimizer_steps']==0
    assert state_digest(b.banks['train'])==old_bank  # no same-Outer switch
    record=deepcopy(b.state['dynamic_banks']['5'])
    restored=StateOnlyBackend();j2=StageJournal(tmp_path/'journal',{'operator':OPERATOR},restored,resume=True)
    j2.run('outer_004/EMPIRICAL_POOL_COMMIT',lambda:pytest.fail('committed pool resampled'))
    assert restored.state['dynamic_banks']['5']==record
    restored._activate_dynamic(5)
    assert restored.state['active_dynamic_outer']==5
    assert restored.banks['train']['empirical_pool_fingerprint']==record['pool_fingerprint']
    assert restored.banks['validation']=={'fixed':1} and restored.schedules['validation']=={'CRN':'fixed'}
    # An uncommitted retry uses deterministic content-addressed files, not a new draw.
    retry=StateOnlyBackend();retry._pool_commit(4)
    assert retry.state['dynamic_banks']['5']==record


def test_source_stage_is_forward_only_and_keeps_models_rng(tmp_path,monkeypatch):
    import analysis.run_finite_direct_large_continuation as runner
    from analysis.nested_equilibrium_utils import atomic_write_json,capture_rng_state
    b=ContinuationBackend.__new__(ContinuationBackend)
    macro=torch.nn.Module();macro.fc1_model=torch.nn.Module()
    macro.fc1_model.FC1_C=torch.nn.Linear(4,1);macro.fc1_model.FC1_K=torch.nn.Linear(4,1)
    macro.sdf_model=torch.nn.Linear(2,1);macro.value_model=torch.nn.Linear(3,1)
    b.models={'policy_value':torch.nn.Linear(7,2),'sdf_fc1':macro}
    b.root=tmp_path;b.args=SimpleNamespace(parent_batch_size=2)
    b.episode=SimpleNamespace(device=torch.device('cpu'))
    b.banks={'train':'train bank'};b.schedules={'train':'train schedule'};b.signature={};b.state={}
    before=state_digest({k:m.state_dict() for k,m in b.models.items()});rng=state_digest(capture_rng_state())
    def fake_forward(models,episode,bank,schedule,signature,path,**kwargs):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        assert bank=='train bank' and schedule=='train schedule' and kwargs['predecision_templates']
        assert not kwargs['capture'];path.mkdir(parents=True)
        atomic_write_json(path/'manifest.json',dict(captures=2));torch.rand(3)
        return None,None,dict(captures=2)
    monkeypatch.setattr(runner,'generate_dataset',fake_forward)
    result=b._empirical_source(4)
    assert result['optimizer_steps']==0 and result['predecision_template_count']==2
    assert before==state_digest({k:m.state_dict() for k,m in b.models.items()})
    assert rng==state_digest(capture_rng_state())


def test_real_runner_order_and_same_outer_bank_activation():
    sequence=list(stages())
    for outer in range(4,12):
        kinds=[k for o,k,i in sequence if o==outer]
        assert kinds.index('firm_bank')<kinds.index('FC1')<kinds.index('SDF')
        assert kinds.index('post_sdf_validation')<kinds.index('empirical_source')<kinds.index('EMPIRICAL_POOL_COMMIT')<kinds.index('cross_block')<kinds.index('outer_commit')
    assert '_activate_dynamic(outer)' in inspect.getsource(ContinuationBackend._execute)
    assert 'self._activate_dynamic' not in inspect.getsource(ContinuationBackend._pool_commit)
    assert 'self.args' in inspect.getsource(ContinuationBackend._empirical_source)
    assert training_plan()['operator']==OPERATOR and training_plan()['source_result_limit'].endswith('NOT_CONVERGENCE_EVIDENCE')


def test_old_cache_and_old_approval_cannot_run():
    with pytest.raises(RuntimeError):verify_cache({'operator':'finite_direct_production_ledger_v1'},{'operator':OPERATOR})
    with pytest.raises(ValueError):require_approval(dict(approved=True,schema_version=3),dict(schema_version=4))
    with pytest.raises(ValueError):require_approval(dict(approved=False,schema_version=4),dict(schema_version=4))


def test_cache_binds_pool_even_if_parent_state_hash_unchanged(monkeypatch):
    import data.finite_direct_runtime as runtime
    monkeypatch.setattr(runtime,'hashes',lambda models:{'firm':'f','sdf':'s','wealth':'w','fc1_c':'c','fc1_k':'k'})
    schedule={'parents':[dict(child_shock_keys=[0,1])]}
    bank=dict(fingerprint='same numerical b',empirical_pool_fingerprint='pool5',bank_version='outer_005')
    before=runtime.cache_provenance({},bank,schedule,{'fingerprint':'initial'},outer=5,stage='D0',economic={})
    bank.update(empirical_pool_fingerprint='other pool')
    after=runtime.cache_provenance({},bank,schedule,{'fingerprint':'initial'},outer=5,stage='D0',economic={})
    with pytest.raises(RuntimeError):verify_cache(before,after)


def test_final_outer_saves_stats_without_constructing_bank12(tmp_path,monkeypatch):
    import analysis.run_finite_direct_large_continuation as runner
    from data.finite_direct_sampling import file_hash
    from analysis.nested_equilibrium_utils import atomic_torch_save,atomic_write_json
    bank,schedule,p=synthetic();p['target_outer']=11;p['fingerprint']=fingerprint(p)
    b,s,h=deepcopy(bank),deepcopy(schedule),[]
    # Fixture bank identity for a committed Outer11; no resampling involved.
    from data.finite_direct_sampling import state_hash
    s.update(empirical_pool_fingerprint=p['fingerprint'],bank_version='outer_011');s['fingerprint']=fingerprint(s)
    b.update(fingerprint=state_hash(b['state']),schedule_fingerprint=s['fingerprint'],empirical_pool_fingerprint=p['fingerprint'],bank_version='outer_011')
    record=save_bundle(tmp_path/'bank11',p,b,s,h)
    path=tmp_path/'source';path.mkdir();atomic_torch_save(path/'batch.pt',dict(captures=source_captures(11)))
    atomic_write_json(path/'manifest.json',dict(shards={'batch.pt':file_hash(path/'batch.pt')}))
    backend=ContinuationBackend.__new__(ContinuationBackend);backend.root=tmp_path/'run'
    backend.state=dict(dynamic_banks={'11':record},dynamic_sources={'11':dict(path=str(path),manifest_sha256=file_hash(path/'manifest.json'),model_hashes={'firm':'f11'})})
    monkeypatch.setattr(runner,'resample_train',lambda *a,**k:pytest.fail('must not generate Bank12'))
    result=backend._pool_commit(11)
    assert result['optimizer_steps']==0 and '12' not in backend.state['dynamic_banks']
    assert (backend.root/'dynamic/final_empirical_pool_statistics.json').exists()


def test_v4_content_and_algorithm_bindings(tmp_path):
    root=tmp_path
    for name in ('initial_dynamic_pool.pt','train_initial_bank.pt','validation_initial_bank.pt','initial.pt'):(root/name).write_bytes(name.encode())
    (root/'sampling_signature.json').write_text(json.dumps(dict(fingerprint='sample')))
    (root/'manifest.json').write_text(json.dumps(dict(source_terminal_checkpoint_sha256='old')))
    def bind():return approval_binding(training_plan(),'commit',root,root/'initial.pt',root/'manifest.json')
    expected=bind();approval=dict(expected,approved=True);require_approval(approval,expected)
    for name in ('initial_dynamic_pool.pt','train_initial_bank.pt','validation_initial_bank.pt','initial.pt'):
        path=root/name;original=path.read_bytes();path.write_bytes(original+b'changed')
        with pytest.raises(ValueError):require_approval(approval,bind())
        path.write_bytes(original)
    changed=dict(expected,dynamic_algorithm_version='changed')
    with pytest.raises(ValueError):require_approval(approval,changed)
