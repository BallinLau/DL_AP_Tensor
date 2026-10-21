"""CPU contracts only; no model optimizer is stepped."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from analysis.h50_source_certification import save_bank_reference,load_bank_reference,certify
from analysis.nested_equilibrium_utils import atomic_torch_save
from data.finite_direct_sampling import state_hash
from data.finite_direct_long_horizon_sampling import bounded_quotas,sample_macro_rows
from data.finite_direct_long_horizon_runtime import continuable_nodes
from test_initial_hstar_h50 import fixture_store


def test_immutable_bank_reference_integrity_and_outer(tmp_path):
    state={'b':torch.tensor([[.1,.2]])};bank=dict(state=state,fingerprint=state_hash(state),bank_version='outer_011')
    ref=save_bank_reference(tmp_path,bank,{'split':'train'},outer=12,pool_version='pool11')
    again=save_bank_reference(tmp_path,bank,{'split':'train'},outer=12,pool_version='pool11')
    assert ref==again and load_bank_reference(ref)[0]['fingerprint']==bank['fingerprint']
    wrong=dict(ref,outer=13)
    with pytest.raises(ValueError,match='Outer/pool'):load_bank_reference(wrong)
    from pathlib import Path
    Path(ref['path']).write_bytes(b'corrupt fixture')
    with pytest.raises(ValueError,match='SHA'):load_bank_reference(ref)


def test_bounded_quota_redistributes_without_inventing_middle_late():
    assert bounded_quotas({0:100,1:1,2:0},10)=={0:9,1:1,2:0}
    assert bounded_quotas({0:2,1:0,2:0},10)=={0:2,1:0,2:0}
    assert sum(bounded_quotas({0:100,1:100,2:100},10).values())==10


def test_macro_sampling_unique_budget_and_pair_identity(tmp_path):
    store=fixture_store(tmp_path)
    rows,audit=sample_macro_rows(store,pair_budget=100,seed=4,complete_pairs_only=True)
    assert audit['unused_quota']==94
    assert audit['actual_parent_times']==6
    assert audit['actual_unique_child_rows']==12
    assert not rows.duplicated(['split','parent_candidate_id','t','child_branch']).any()
    assert all(set(g.child_branch)=={0,1} for _,g in rows.groupby(['split','parent_candidate_id','t']))


def test_zero_hard_survivors_does_not_preempt_original_entry():
    s=dict(alive=torch.zeros(1,2,dtype=torch.bool),hatc_cal=torch.zeros(1),lnk_cal=torch.zeros(1))
    rec=dict(C_raw_t1=1.,K_cal_t1=2.,Hatc_cal_t1=-.69,LnK_cal_t1=.69,economic_state_valid=False)
    assert continuable_nodes(s,[rec])==[0]
    for key in ('C_raw_t1','K_cal_t1'):
        bad=dict(rec,**{key:float('nan')})
        assert continuable_nodes(s,[bad])==[]


def test_wrong_source_job_rejected_before_reading():
    with pytest.raises(ValueError,match='Job'):certify(None,None,None,source_job=1)


def test_snapshot_stores_only_bound_references(monkeypatch,tmp_path):
    import analysis.run_initial_hstar_h50_broad_macro as m
    monkeypatch.setattr(m.ProductionBackend,'snapshot',lambda self:dict(state={}))
    obj=object.__new__(m.H50Backend);obj._active_bank_ref={'sha256':'active'};obj._source_bank_ref={'sha256':'source'}
    snap=obj.snapshot()
    assert 'h50_train_bank' not in snap
    assert snap['h50_train_bank_reference']['sha256']=='active'


@pytest.mark.parametrize('fault',['none','stage','commit','operator','export','active_outer','optimizer','checkpoint_sha'])
def test_committed_source_certification_contract(monkeypatch,tmp_path,fault):
    import analysis.h50_source_certification as m
    tx=dict(models={k:{'w':torch.ones(1)} for k in ('policy_value','sdf_fc1','fc2')},
        optimizers={k:dict(state={0:torch.ones(1)},param_groups=[{}]) for k in ('policy_value','sdf_fc1','fc1_conditional','fc2')},
        episode_runtime=dict(lr_schedulers={'fc1':{'step':1}},sdf_al_state={'rho':2}),
        rng_state=dict(python=1,numpy=2,torch=torch.ones(1),cuda=[]))
    record={k:dict(path=k,sha256=k) for k in ('pool','bank','schedule')}
    record.update(outer=11)
    learning=dict(transaction=tx,fresh=dict(pair_rng=torch.ones(1),bank={}),
        state=dict(active_dynamic_outer=11,dynamic_banks={'11':record}))
    journal=dict(stage=m.SOURCE_STAGE,completed=[f'outer_{o:03d}/outer_commit' for o in range(4,12)],
        contract=dict(source_commit=m.SOURCE_COMMIT,approval_binding=dict(economic_operator_version=m.OPERATOR)),
        backend=deepcopy(learning))
    if fault=='stage':journal['stage']='outer_010/outer_commit'
    if fault=='commit':journal['contract']['source_commit']='wrong'
    if fault=='operator':journal['contract']['approval_binding']['economic_operator_version']='v1'
    if fault=='export':learning['fresh']['bank']={'different':1}
    if fault=='active_outer':learning['state']['active_dynamic_outer']=12;journal['backend']=deepcopy(learning)
    if fault=='optimizer':learning['transaction']['optimizers'].pop('fc1_conditional');journal['backend']=deepcopy(learning)
    pointer=tmp_path/'latest.json';pointer.write_text('{"checkpoint":"checkpoint.pt","sha256":"journal"}')
    monkeypatch.setattr(m,'safe_load',lambda p:learning if p=='export' else journal)
    monkeypatch.setattr(m,'file_hash',lambda p:('bad' if fault=='checkpoint_sha' else m.SOURCE_SHA) if p=='export' else 'journal')
    monkeypatch.setattr(m,'load_bundle',lambda r:(dict(fingerprint='pool'),dict(bank_version='outer_011',fingerprint='bank'),dict(split='train')))
    if fault=='none':assert m.certify('export','journal',pointer,source_job=702636)[-1]['source_checkpoint_audit']['verified']
    else:
        with pytest.raises(ValueError):m.certify('export','journal',pointer,source_job=702636)


def test_stagejournal_restores_reference_and_does_not_repeat_commit(tmp_path):
    from analysis.finite_direct_outer4_state import StageJournal
    state={'b':torch.tensor([[.1,.2]])};bank=dict(state=state,fingerprint=state_hash(state),bank_version='outer_011')
    ref=save_bank_reference(tmp_path,bank,{'split':'train'},outer=12,pool_version='pool11')
    class Backend:
        def snapshot(self):return {'bank_reference':ref}
        def restore(self,snapshot):self.bank,self.schedule=load_bank_reference(snapshot['bank_reference'])
    a=Backend();journal=StageJournal(tmp_path/'journal',{'review':'v2'},a)
    journal.run('outer12/committed',lambda:{'optimizer_steps':0})
    b=Backend();resumed=StageJournal(tmp_path/'journal',{'review':'v2'},b,resume=True)
    assert resumed.run('outer12/committed',lambda:pytest.fail('repeated committed stage'))=={'optimizer_steps':0}
    assert b.bank['fingerprint']==bank['fingerprint']


def test_frozen_cpu_does_not_reinterpret_cuda_pair_rng(monkeypatch):
    from analysis.run_h50_frozen_extinction_review import frozen_cpu_fresh_state
    import analysis.run_full_nested_finite_direct_outer4 as runner
    seen=[];monkeypatch.setattr(runner,'restore_fresh',lambda e,s:seen.append(s))
    source={'pair_rng':torch.ones(16,dtype=torch.uint8),'bank':None}
    audit=frozen_cpu_fresh_state(None,source)
    assert seen[0]['pair_rng'] is None and source['pair_rng'].numel()==16
    assert audit['source_cuda_rng_preserved_not_converted']
    assert not audit['sdf_pair_rng_used_in_frozen_rollout']


def test_production_fixed_validation_matches_certified_source(monkeypatch,tmp_path):
    import analysis.h50_source_certification as m
    expected={'validation_initial_bank.pt':m.FIXED_VALIDATION_SHA,
        'validation_parent_sampling_schedule.json':m.FIXED_VALIDATION_SCHEDULE_SHA}
    monkeypatch.setattr(m,'file_hash',lambda p:expected[p.name])
    m.verify_fixed_validation(tmp_path)
    for key in list(expected):
        original=expected[key];expected[key]='changed'
        with pytest.raises(ValueError,match='Validation changed'):m.verify_fixed_validation(tmp_path)
        expected[key]=original
