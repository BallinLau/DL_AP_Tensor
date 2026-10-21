"""Non-optimizing formal execution contracts."""
from types import SimpleNamespace
from copy import deepcopy
import pandas as pd
import pytest
import torch
from analysis.run_initial_hstar_h50_broad_macro import plan,H50Backend,distinct_balanced_donors
from data.finite_direct_long_horizon_sampling import sample_macro_rows
from test_h50_coverage_round2 import donor


def test_nine_full_outer_schedule():
    p=plan(end_outer=20,skip_bp_head=True)
    assert p['end_outer']==20 and p['bp_head_training_skipped_by_user']
    for o in range(12,21):
        kinds=[s['kind'] for s in p['stages'] if s['outer']==o]
        assert kinds.count('P')==9 and kinds.count('Q')==9 and kinds.count('BP')==3
        assert kinds.count('SDF')==2 and kinds.count('data')==3
        assert kinds.index('EMPIRICAL_POOL_COMMIT')<kinds.index('outer_commit')
    assert len([s for s in p['stages'] if s['kind']=='outer_commit'])==9


def test_bp_skip_has_no_optimizer_or_model_mutation(tmp_path):
    b=object.__new__(H50Backend);b.root=tmp_path;b.args=SimpleNamespace(skip_bp_head=True)
    b.models={'policy_value':torch.nn.Linear(7,2)}
    b.episode=SimpleNamespace(hyperparams=SimpleNamespace(simulation_bp_action_source='grid'))
    before=deepcopy(b.models['policy_value'].state_dict())
    result=b._execute(12,'BP',{'inner':0})
    assert result['optimizer_steps']==0 and result['BP_HEAD_TRAINING_SKIPPED_BY_USER']
    for k,v in before.items():assert torch.equal(v,b.models['policy_value'].state_dict()[k])
    b.episode.hyperparams.simulation_bp_action_source='head'
    with pytest.raises(ValueError,match='Grid'):b._execute(12,'BP',{})


def test_empty_pairs_explicit_skip_preserves_state(tmp_path):
    b=object.__new__(H50Backend);b._data=lambda *a:pd.DataFrame()
    b.state={'sdf_epoch':123,'al':{'dual':1.25}}
    before=deepcopy(b.state)
    assert b._sdf_chunk(12,0)['status']=='SDF_SKIPPED_NO_VALID_PAIRS'
    assert b.state==before and not b._sdf_sampling
    store=SimpleNamespace(rows=lambda:iter([pd.DataFrame()]))
    # Empty typed fixture avoids inventing a valid economic label.
    store.rows=lambda:iter([])
    frame,index=sample_macro_rows(store,pair_budget=8192,seed=1,complete_pairs_only=True,allow_empty_pairs=True)
    assert frame.empty and index['actual_parent_times']==0
    with pytest.raises(ValueError,match='NO_COMPLETE'):sample_macro_rows(store,pair_budget=8192,seed=1,complete_pairs_only=True)


def test_donor_capacity_not_filled_by_replacement():
    b=donor();n=len(b.parent_states)
    for k in ('path','source_time','source_branch','firm_id'):
        b.metadata[k]=torch.arange(n)//2
    # Duplicate draws have the same provenance; they cannot inflate capacity.
    u=distinct_balanced_donors(b)
    assert len(u.parent_states)==n//2 and u.metadata['donor_shortfall']==n//2
    assert u.metadata['unique_source_firm_time_count']==n//2
