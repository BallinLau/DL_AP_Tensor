"""Real production forward-only loss and simulator interface tests."""
import math
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
import torch
from config import Config
from experiments.run_utils import build_models,build_hyperparams,build_optimizers
from training.episode import Episode,SDFTrainingPhase
from analysis.run_full_nested_finite_direct_outer4 import sdf_batches,set_scope
from analysis.run_fc1_self_consistent_macro import production_loss
from data.finite_direct_runtime import cache_provenance,verify_cache,generate_dataset
from data.finite_direct_sampling import construct_parent_bank,fixed_schedule,SamplingConfig,template_fingerprint,state_hash
from analysis.nested_equilibrium_utils import stable_json_hash


@pytest.fixture
def production(monkeypatch):
    def forbidden(*a,**k): raise AssertionError('production optimization forbidden in review tests')
    for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.LBFGS): monkeypatch.setattr(cls,'step',forbidden)
    Config.DEVICE=torch.device('cpu'); hp=build_hyperparams(); models=build_models(torch.device('cpu'))
    hp.entry_mode='investment_compare'; hp.entry_spec_version='investment_compare_v1'; hp.entry_capital_ratio=.05
    hp.entry_size_ratio=1.; hp.entry_cost_max=.5; hp.consumption_aggregation_mode='raw'; hp.consumption_floor=0.
    hp.node_accounting_mode='economic_node_ledger'; hp.market_clearing_mode='disabled'
    hp.simulation_bp_action_source='grid'; hp.fc1_recursive_aux_training_enabled=False
    e=Episode(models=models,optimizers=build_optimizers(models,hp),config=Config,hyperparams=hp,
        device=torch.device('cpu'),episode_id=1,q_checkpoint_loaded=False)
    e._fc1_firms=200
    return models,e


def frame():
    return pd.DataFrame([dict(split='train',parent_candidate_id=p,t=1,child_branch=b,
        economic_state_valid=True,x_t=.01,x_t1=.02+b*.03,Hatc_cal_t=-1.,LnK_cal_t=4.,
        Hatc_cal_t1=math.log(20/60),LnK_cal_t1=math.log(60),C_raw_t1=20.,K_cal_t1=60.,
        Hatcf_t=-2.,LnKF_t=3.8,Hatcf_t1=-1.1,LnKF_t1=4.05) for p in range(4) for b in (0,1)])


def test_actual_sdf_true_only_consumes_true_parent_and_not_dummy_M(production):
    models,e=production; e.add_FC1loss=False; e.set_sdf_training_phase(SDFTrainingPhase.SDF_TRUE_ONLY)
    set_scope(models,'sdf'); batches,_=sdf_batches(frame(),128,'cpu')
    e._prepare_sdf_shock_bank_for_epoch(batches,0,['sdf_fc1'])
    model=models['sdf_fc1']; original=model.forward_step; observed=[]
    def spy(**kw):
        observed.append((kw['hatcf_prev'].clone(),kw['lnkf_prev'].clone())); return original(**kw)
    model.forward_step=spy
    pair_rng=e._sdf_pair_generator.get_state().clone()
    with torch.no_grad(): first=e._compute_sdf_loss(batches[0])
    assert torch.isfinite(first)
    torch.testing.assert_close(observed[0][0],torch.full((4,1),-1.))
    torch.testing.assert_close(observed[0][1],torch.full((4,1),4.))
    changed=deepcopy(batches[0]); changed['parent'][:,5:7]=float('nan')
    for c in changed['children']: c[:,7]=float('nan')
    e._sdf_pair_generator.set_state(pair_rng)
    with torch.no_grad(): second=e._compute_sdf_loss(changed)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    assert e.sdf_fc1_step_count==0
    # Exercise actual pooled AL estimator (read-only) as well as wealth loss.
    constraints=e._estimate_sdf_al_constraints(batches)
    assert all(np.isfinite(constraints[k]) for k in ('g_mean_low','g_mean_high','g_var_high'))


def test_actual_fc1_single_child_loss(production):
    models,e=production; e.set_sdf_training_phase(SDFTrainingPhase.FC1_ONLY)
    X=torch.tensor([[.01,.02,-1.,4.]]); Y=torch.tensor([[-.9,4.1]])
    with torch.no_grad(): loss=production_loss(e,X,Y)
    assert torch.isfinite(loss) and e.sdf_fc1_step_count==0


def test_stale_model_cache_rejected(production):
    models,_=production; bank=dict(fingerprint='bank'); schedule={'parents':[{'child_shock_keys':[0,1]}]}
    kw=dict(outer=1,stage='D0',economic={'test':'fixed'})
    a=cache_provenance(models,bank,schedule,{'fingerprint':'sampling'},**kw)
    with torch.no_grad(): next(models['policy_value'].parameters()).add_(.01)
    b=cache_provenance(models,bank,schedule,{'fingerprint':'sampling'},**kw)
    with pytest.raises(RuntimeError,match='stale'): verify_cache(a,b)
    a=b
    with torch.no_grad(): next(models['sdf_fc1'].fc1_model.parameters()).add_(.01)
    b=cache_provenance(models,bank,schedule,{'fingerprint':'sampling'},**kw)
    with pytest.raises(RuntimeError,match='stale'): verify_cache(a,b)


def test_no_pair_is_not_legal_sdf_batch():
    f=frame(); f=f[f.child_branch==0]
    with pytest.raises(ValueError,match='no complete'): sdf_batches(f,128,'cpu')


def test_committed_firm_block_refreshes_actual_grid_target(production):
    from analysis.run_full_nested_finite_direct_outer4 import ProductionBackend
    from analysis.nested_equilibrium_utils import module_hash
    models,e=production; backend=object.__new__(ProductionBackend)
    backend.models=models; backend.episode=e; backend.fc2_guard={'calls':0}; backend.counter={'ROOT_SEARCH_CALLS':0}
    def fake_update(*args):
        # No optimizer: deliberate unit-fixture mutation tests target routing.
        with torch.no_grad(): next(models['policy_value'].parameters()).add_(.001)
        return {'fake_update':True}
    backend._execute=fake_update
    backend.execute(1,'P',{'inner':0,'policy':0})
    assert module_hash(e._target_policy_value())==module_hash(models['policy_value'])


def test_actual_production_parent_two_children_and_capture(production,monkeypatch,tmp_path):
    from data import simulate_ts_parallel as parallel
    from data.simulate_ts import SimulateTS
    from analysis.run_fc1_self_consistent_macro import simulator
    from analysis.nested_equilibrium_utils import tensor_hash
    from data.finite_direct_runtime import capture_node
    # Synthetic policy fixture, real batched production ledger/entry/transition.
    # This checks adapter wiring, not economic results of trained networks.
    models,e=production; e._fc1_firms=4
    from training.bp_simulation_policy import GridBPSimulationPolicy
    # Production refuses CPU grid for formal runs. Only this tiny interface
    # test opts into the class's existing CPU reference option; runner unchanged.
    monkeypatch.setattr(e,'_build_simulation_bp_grid_policy',lambda:GridBPSimulationPolicy(
        target_model=models['policy_value'],sdf_fc1_model=models['sdf_fc1'],
        p0_loss=e.loss_fns['p0'],pi_loss=e.loss_fns['pi'],hyperparams=e.hyperparams,
        economic_config=Config,n_child_shocks=2,shock_seed=123,require_cuda=False))
    def fixed_policy(model,state):
        zero=torch.zeros((len(state),1)); bp=zero+.2
        return SimpleNamespace(Q=zero,P0=zero,PI=zero,bar_i=zero,bar_z=zero,P=zero,
                               bp0=bp,bpI=bp,bp=bp)
    monkeypatch.setattr(parallel,'forward_policy_value_for_simulation',fixed_policy)
    sim=simulator(models,e,2,20261020,device=torch.device('cpu'))
    state=parallel._initialize_batched_state(sim,4)
    state['x'].fill_(2.); state['K'].fill_(1.); state['lnkf'].fill_(math.log(4))
    state['transition_path_id']=torch.tensor([2000000,2000001])
    bank={'state':state,'fingerprint':state_hash(state)}
    schedule=dict(split='train',seed=20261020,parents=[dict(child_shock_keys=[dict(branch=b,path_id=p) for b in (0,1)])
                                                        for p in [2000000,2000001]])
    f,snaps,m=generate_dataset(models,e,bank,schedule,{'fingerprint':'test'},tmp_path,
                              outer=1,stage='D0',parent_batch_size=2)
    assert len(f)==4 and set(f.child_branch)=={0,1}
    assert len(snaps)==3 and {s['source_time'] for s in snaps}=={0,1}
    assert not f.duplicated(['parent_candidate_id','child_branch']).any()
    assert set(f.parent_candidate_id)=={2000000,2000001}
    assert f.economic_state_valid.all()
    for pid,g in f.groupby('parent_candidate_id'): assert g.x_t1.nunique()==2
    # Reuse is byte/hash verified and causes no additional policy invocation.
    def no_forward(*a,**k): raise AssertionError('cache must not simulate again')
    monkeypatch.setattr(parallel,'forward_policy_value_for_simulation',no_forward)
    repeat,_,_=generate_dataset(models,e,bank,schedule,{'fingerprint':'test'},tmp_path,
                                 outer=1,stage='D0',parent_batch_size=2)
    pd.testing.assert_frame_equal(f,repeat)
