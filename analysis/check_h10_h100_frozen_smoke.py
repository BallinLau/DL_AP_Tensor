"""Bounded real-checkpoint CPU/GPU replay smoke; no optimizer invocation."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import sys,time,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from analysis.finite_direct_continuation_state import safe_load
from analysis.run_fc1_self_consistent_macro import setup,select_state,hashes
from analysis.nested_equilibrium_utils import atomic_write_json
from data.finite_direct_sampling import state_hash,file_hash
from data.finite_direct_replay_runtime import generate_long_horizon_dataset
from data.finite_direct_longrun_replay import commit_pool,load_pool,select_mixture,fit_edges,eligibility


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--schedule',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    torch.set_num_threads(4)
    cp=safe_load(a.checkpoint)
    models,e,_,_=setup(SimpleNamespace(device=a.device),initial_models=cp['transaction']['models'])
    e._fc1_firms=200;e.refresh_firm_target();before=hashes(models)
    if a.device=='cpu':
        # Explicit diagnostic-only CPU execution of the SAME Grid teacher.
        # Production CUDA requirement is untouched, never a head fallback.
        from training.bp_simulation_policy import GridBPSimulationPolicy
        e._build_simulation_bp_grid_policy=lambda:GridBPSimulationPolicy(
            target_model=e._target_policy_value(),sdf_fc1_model=models['sdf_fc1'],
            p0_loss=e.loss_fns['p0'],pi_loss=e.loss_fns['pi'],hyperparams=e.hyperparams,
            economic_config=e.config,n_child_shocks=e.hyperparams.simulation_bp_grid_n_child_shocks,
            shock_seed=getattr(e.hyperparams,'simulation_bp_grid_shock_seed',314159),require_cuda=False)
    for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.LBFGS):
        cls.step=lambda *a,**k:(_ for _ in ()).throw(AssertionError('frozen smoke optimizer forbidden'))
    bank=safe_load(a.bank);schedule=safe_load(a.schedule)
    state=select_state(bank['state'],slice(0,4),torch.device('cpu'))
    bank=dict(bank,state=state,fingerprint=state_hash(state))
    schedule=dict(schedule,parents=schedule['parents'][:4])
    short=generate_long_horizon_dataset(models,e,bank,schedule,{'fingerprint':'FROZEN_SMOKE_ONLY'},a.output/'h10',
        outer=12,stage='FROZEN_SMOKE',horizon=10,branches=2,capture=False,parent_batch_size=4)
    long=generate_long_horizon_dataset(models,e,bank,schedule,{'fingerprint':'FROZEN_SMOKE_ONLY'},a.output/'h100',
        outer=12,stage='FROZEN_SMOKE',horizon=100,branches=1,capture=False,parent_batch_size=4)
    import pandas as pd
    coverage=pd.concat([eligibility(f).fc1 for f in short.rows()],ignore_index=True)
    pool=commit_pool([long],a.output/'pool',source_outer=12,model_hashes=dict(firm=before['firm'],fc1=before['fc1_c']+before['fc1_k'],sdf=before['sdf']),
        initial_design=[dict(smoke_only=True,paths=4)],outer_checkpoint_sha=file_hash(a.checkpoint))
    replay=load_pool(a.output/'pool',consumer_outer=13)
    selected,audit=select_mixture(coverage,replay,edges=fit_edges([coverage,replay]),target=16)
    assert hashes(models)==before
    atomic_write_json(a.output/'summary.json',dict(status='FROZEN_REPLAY_PASS',training_performed=False,
        full_optimizer_pipeline_smoke=False,checkpoint_sha256=file_hash(a.checkpoint),
        paths=4,scheduled_h10=short.manifest['scheduled_child_rows'],
        scheduled_h100=long.manifest['scheduled_child_rows'],longrun_pool=pool,
        fc1_mixture=audit,selected_rows=len(selected),models_unchanged=True,
        elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(status='FROZEN_REPLAY_PASS',pool=pool['status'],seconds=time.monotonic()-started)))


if __name__=='__main__':main()
