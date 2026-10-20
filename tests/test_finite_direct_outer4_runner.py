"""Fake-optimizer orchestration; never run production optimization."""
from dataclasses import replace
import json
from types import SimpleNamespace
import pytest
import torch
from analysis.run_full_nested_finite_direct_outer4 import (Budget,stages,stage_id,drive,
    training_plan,parse_args,require_approval,set_scope)
from analysis.finite_direct_outer4_state import StageJournal


class FakeBackend:
    def __init__(self): self.state={'value':0,'optimizer':0,'AL':2.,'dataset':'D0'}; self.calls=[]
    def snapshot(self): return dict(state=dict(self.state),rng=torch.get_rng_state().clone())
    def restore(self,s): self.state=dict(s['state']); torch.set_rng_state(s['rng'])
    def execute(self,o,k,i):
        self.calls.append((o,k,dict(i))); self.state['value']+=1
        if k in ('P','Q','BP','FC1','SDF'): self.state['optimizer']+=1
        return dict(fake=True,kind=k,outer=o)


def test_real_drive_four_outer_order_and_no_outer0_firm_steps(tmp_path):
    b=FakeBackend(); j=StageJournal(tmp_path,{'plan':'test'},b)
    drive(b,j)
    assert {o for o,k,i in b.calls}=={0,1,2,3}
    assert not any(o==0 and k in ('P','Q','BP','firm_bank') for o,k,i in b.calls)
    for o in range(4):
        calls=[(k,i) for n,k,i in b.calls if n==o]
        macro=[i['version'] for k,i in calls if k=='data']; assert macro==[0,1,2]
        assert [i['chunk'] for k,i in calls if k=='SDF']==[0,1]
        assert calls[-1][0]=='outer_commit'
    first=len(b.calls); drive(b,j); assert len(b.calls)==first


def test_cli_defaults_and_approval_boundary(tmp_path):
    args=parse_args(['--sampling-dir','sampling','--initial-checkpoint','init.pt','--output','out'])
    assert args.parent_batch_size==128 and not args.execute_approved
    assert require_approval(args,training_plan(),'commit') is False
    args.execute_approved=True
    with pytest.raises(RuntimeError,match='approval'): require_approval(args,training_plan(),'commit')
    path=tmp_path/'approval.json'; path.write_text(json.dumps(dict(approved=True,source_commit='wrong')))
    args.approval=path
    with pytest.raises(RuntimeError,match='exact'): require_approval(args,training_plan(),'commit')


def test_freeze_scope():
    macro=torch.nn.Module(); macro.fc1_model=torch.nn.Linear(4,2)
    macro.sdf_model=torch.nn.Linear(4,1); macro.value_model=torch.nn.Linear(3,1)
    models=dict(policy_value=torch.nn.Linear(7,2),sdf_fc1=macro,fc2=torch.nn.Linear(1,1))
    set_scope(models,'sdf')
    assert not any(p.requires_grad for p in models['policy_value'].parameters())
    assert not any(p.requires_grad for p in macro.fc1_model.parameters())
    assert all(p.requires_grad for p in macro.value_model.parameters())
    assert not any(p.requires_grad for p in models['fc2'].parameters())


def test_budget_locked():
    budget=training_plan()['budget']
    assert budget['equilibrium_outer_max']==4 and budget['firm_inner_max']==3
    assert budget['fc1_min_epochs']==25 and budget['fc1_max_epochs']==40
    assert budget['sdf_chunks']*budget['sdf_epochs_per_chunk']==100
