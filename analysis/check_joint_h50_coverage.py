"""Small CPU frozen coverage probe; explicitly bounded, never training."""
import argparse,json,sys,time,resource,platform,subprocess
from pathlib import Path
from dataclasses import replace,asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import pandas as pd
from analysis.h50_source_certification import certify,verify_fixed_validation
from analysis.run_fc1_self_consistent_macro import setup,simulator,select_state,hashes
from analysis.run_h50_frozen_extinction_review import frozen_cpu_fresh_state
from analysis.nested_equilibrium_utils import restore_training_transaction,atomic_write_json
from analysis.economic_config import AnalysisEconomicConfig
from data.finite_direct_joint_coverage import JointCoverageConfig,build_joint_bank
from data.finite_direct_long_horizon_runtime import generate_long_horizon_dataset
from data.finite_direct_sampling import state_hash,file_hash
from analysis.h50_joint_diagnostics import write_dashboard,compare_validation_stages
from training.bp_simulation_policy import GridBPSimulationPolicy


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidates',type=int,default=64);p.add_argument('--parents',type=int,default=16)
    p.add_argument('--max-rollout-seconds',type=float,default=600.)
    a=p.parse_args()
    if not 4<=a.candidates<=64 or not 1<=a.parents<=32:p.error('explicit frozen budget only: <=64 candidates / <=32 Parents')
    if a.output.exists():raise ValueError('new frozen output directory required; preserve previous results')
    a.output.mkdir(parents=True);torch.set_num_threads(4)
    def forbidden(*args,**kwargs):raise RuntimeError('optimization forbidden in frozen probe')
    for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.LBFGS):cls.step=forbidden
    torch.Tensor.backward=forbidden
    started=time.monotonic();r=a.inputs
    learning,reference,schedule,pool,audits=certify(r/'outer_011_complete.pt',r/'source_journal_00907.pt',r/'latest.json',source_job=702636,artifact_dir=r)
    verify_fixed_validation(r)
    for k,v in audits.items():atomic_write_json(a.output/(k+'.json'),v)
    models,e,fc1,guard=setup(argparse.Namespace(device='cpu'),initial_models=learning['transaction']['models'])
    restore_training_transaction(learning['transaction'],models=models,optimizers={**e.optimizers,'fc1_conditional':fc1},episode=e)
    fresh=frozen_cpu_fresh_state(e,learning['fresh']);e._fc1_firms=200;e.refresh_firm_target()
    e._build_simulation_bp_grid_policy=lambda:GridBPSimulationPolicy(target_model=e._target_policy_value(),sdf_fc1_model=models['sdf_fc1'],
        p0_loss=e.loss_fns['p0'],pi_loss=e.loss_fns['pi'],hyperparams=e.hyperparams,economic_config=e.config,
        n_child_shocks=e.hyperparams.simulation_bp_grid_n_child_shocks,shock_seed=getattr(e.hyperparams,'simulation_bp_grid_shock_seed',314159),require_cuda=False)
    before=hashes(models);econ=dict(config=asdict(AnalysisEconomicConfig.from_current_config()),hyperparameters=asdict(e.hyperparams))
    cfg=replace(JointCoverageConfig(),max_candidates=a.candidates,candidate_batch_size=4,bin_cap=16)
    factory=lambda n:simulator(models,e,n,cfg.seed,device=torch.device('cpu'))
    # Four first, then the separately budgeted coverage batch. Never 8192.
    with torch.inference_mode():
        smoke=build_joint_bank(reference,[],outer=12,target=4,cfg=replace(cfg,max_candidates=4),
            output=a.output/'tiny4',simulator_factory=factory,economic=econ,committed=True)
        bank,plan,stats=build_joint_bank(reference,pool['recent']+pool['archive'],outer=12,target=a.parents,cfg=cfg,
            output=a.output/'coverage',simulator_factory=factory,economic=econ,committed=True,pool_fingerprint=pool['fingerprint'])
        rollout_start=time.monotonic()
        def bounded(*args):
            if time.monotonic()-rollout_start>a.max_rollout_seconds:raise TimeoutError('FROZEN_CPU_BUDGET_REACHED_NOT_ECONOMIC_EXTINCTION')
        status='COMPLETE'
        try:
            store=generate_long_horizon_dataset(models,e,bank,plan,{'fingerprint':'FROZEN_JOINT_COVERAGE_PROBE_V1'},
                a.output/'h50',outer=12,stage='FROZEN_COVERAGE_PROBE',parent_batch_size=4,
                diagnostic_observer=bounded,diagnostic_metadata=dict(initial_bank_sha=file_hash(a.output/'coverage/selected_bank.pt'),coverage_config=asdict(cfg)))
            write_dashboard(store,a.output/'diagnostics',cfg)
        except TimeoutError:
            status='PARTIAL_CPU_BUDGET';store=None
        compare_validation_stages({},a.output/'diagnostics')
    if hashes(models)!=before:raise RuntimeError('frozen model changed')
    raw_bytes=sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())
    valid_rows=0
    if store is not None:
        valid_rows=sum(int(f.economic_state_valid.sum()) for f in store.rows())
    seconds=time.monotonic()-started
    manifest=dict(status=status,source_job=702636,source_outer=11,model_hashes=before,models_unchanged=True,
        proposed_budget=a.candidates,selected_parents=len(bank['state']['x']),target_parents=a.parents,coverage=stats,
        horizon=50,valid_fc1_rows=valid_rows if store else None,training_performed=False,optimizer_steps=0,
        fc2_calls=guard['calls'],fresh_cpu_compatibility=fresh,gpu_performance='NOT_TESTED',trained_convergence='NOT_TESTED',
        total_seconds=seconds,rollout_seconds=time.monotonic()-rollout_start,output_bytes=raw_bytes,
        peak_cpu_memory_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if platform.system()=='Darwin' else 1024**2),
        effective_valid_rows_per_second=valid_rows/seconds if store else None,
        generation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        tracked_dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()),
        code_sha256={name:file_hash(ROOT/name) for name in ('data/finite_direct_joint_coverage.py','data/finite_direct_long_horizon_runtime.py','analysis/h50_joint_diagnostics.py','analysis/check_joint_h50_coverage.py')})
    atomic_write_json(a.output/'probe_summary.json',manifest);print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
