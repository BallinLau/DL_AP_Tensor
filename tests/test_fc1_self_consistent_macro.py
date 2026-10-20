import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.run_fc1_self_consistent_macro import (
    paired_change,coverage,reason,select_state,production_loss,simulate_bank,
)
from data.simulate_ts import SimulateTS
from data import simulate_ts_parallel as parallel
from config import Config


def frame():
    return pd.DataFrame([
        dict(parent_candidate_id=1,child_branch=0,economic_state_valid=True,failure_reason='',normalized_G_at_prediction=.2),
        dict(parent_candidate_id=1,child_branch=1,economic_state_valid=False,failure_reason='NONPOSITIVE_CONSUMPTION',normalized_G_at_prediction=np.nan),
        dict(parent_candidate_id=2,child_branch=0,economic_state_valid=True,failure_reason='',normalized_G_at_prediction=.4),
        dict(parent_candidate_id=2,child_branch=1,economic_state_valid=True,failure_reason='',normalized_G_at_prediction=.3),
    ])


def test_single_child_labels_and_cluster_counts():
    c=coverage(frame(),2)
    assert c['valid_children']==3 and c['valid_parents']==2
    assert c['complete_two_child_parents']==1
    assert c['effective_sample_size_parent_clusters']==2
    assert c['economic_failure_fraction']==.25


def test_matched_valid_does_not_impute_failed_child():
    a=frame(); b=a.copy(); b.loc[0,'economic_state_valid']=False
    b.loc[2:,'normalized_G_at_prediction']*=.5
    m=paired_change(a,b)
    assert len(m)==2
    np.testing.assert_allclose(m.relative_improvement,.5)


def test_failure_classification_and_no_floor():
    assert reason(-1,2,4,float('nan'),.7)=='NONPOSITIVE_CONSUMPTION'
    assert reason(1,0,4,0,float('nan'))=='UNDEFINED_LNK'
    assert reason(1,2,0,0,.7)=='ZERO_SURVIVING_MASS'


def test_slice_owns_storage_and_preserves_identity():
    a={'x':torch.tensor([1.,2.]),'transition_path_id':torch.tensor([91,99]),'b':torch.ones(2,3)}
    b=select_state(a,torch.tensor([1])); b['b'].zero_()
    assert b['transition_path_id'].tolist()==[99]
    assert torch.equal(a['b'],torch.ones(2,3))


def test_production_loss_tensor_layout():
    class E:
        def _compute_fc1_only_loss(self,batch,parent,children):
            torch.testing.assert_close(parent[:,4],X[:,0])
            torch.testing.assert_close(parent[:,7:9],X[:,2:4])
            torch.testing.assert_close(children[:,0,4],X[:,1])
            torch.testing.assert_close(children[:,0,8:10],Y)
            assert children.shape[1]==1
            return torch.tensor(1.)
    X=torch.randn(7,4); Y=torch.randn(7,2)
    assert production_loss(E(),X,Y)==1.


def test_direct_negative_account_retention_is_opt_in(monkeypatch):
    from data.entry import EntryEconomicInfeasibilityError
    class Policy(torch.nn.Module):
        def forward(self,states): pass
    def forward(model,states):
        z=torch.zeros((len(states),1)); o=torch.ones_like(z)
        return SimpleNamespace(Q=z,P0=z,PI=z,bar_i=o,bar_z=z,P=z,bp0=z,bpI=z,bp=z)
    monkeypatch.setattr(parallel,'forward_policy_value_for_simulation',forward)
    sim=SimulateTS(models={'policy_value':Policy(),'sdf_fc1':None},config=Config,n_paths=2,
                   group_size=2,horizon=1,device=torch.device('cpu'),enable_entry=False,
                   consumption_aggregation_mode='raw',consumption_floor=0.)
    state=parallel._initialize_batched_state(sim,2)
    state['x'].fill_(-12); state['i'].fill_(1e3)
    with pytest.raises(EntryEconomicInfeasibilityError):
        parallel._process_node_batched(sim,select_state(state,slice(None)),0,-1)
    sim.diagnostic_retain_undefined_nodes=True
    s=select_state(state,slice(None))
    _,macro=parallel._process_node_batched(sim,s,0,-1)
    c=sim.MACRO_COLUMNS.index('C_raw'); h=sim.MACRO_COLUMNS.index('Hatc')
    assert (macro[:,c]<0).all()
    assert torch.isnan(macro[:,h]).all()


def test_simulation_cache_rejects_changed_fc1(tmp_path):
    import json
    (tmp_path/'simulation_manifest.json').write_text(json.dumps({'wrong':'old model'}))
    model=SimpleNamespace(fc1_model=torch.nn.Linear(4,2))
    with pytest.raises(RuntimeError,match='different FC1'):
        simulate_bank({'sdf_fc1':model},SimpleNamespace(),SimpleNamespace(firms=200),
                      {'x':torch.ones(2)},3,'train',tmp_path)
