"""Four-parent frozen next-Outer Bank construction from real ledger templates."""
import argparse,sys,json,time
from pathlib import Path
from dataclasses import replace,asdict
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from analysis.run_fc1_self_consistent_macro import setup,simulator,hashes
from analysis.finite_direct_continuation_state import safe_load
from analysis.nested_equilibrium_utils import atomic_write_json
from analysis.economic_config import AnalysisEconomicConfig
from analysis.h50_source_certification import verify_fixed_validation
from analysis.h50_joint_diagnostics import write_dashboard
from data.finite_direct_long_horizon_empirical import committed_templates
from data.finite_direct_long_horizon_runtime import generate_long_horizon_dataset
from data.finite_direct_joint_coverage import JointCoverageConfig,build_joint_bank
from data.finite_direct_sampling import file_hash
from training.bp_simulation_policy import GridBPSimulationPolicy


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--source-store',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);started=time.monotonic()
    def no(*args,**kw):raise RuntimeError('frozen pool contract forbids optimization')
    torch.Tensor.backward=no
    for c in (torch.optim.Adam,torch.optim.AdamW,torch.optim.LBFGS):c.step=no
    verify_fixed_validation(a.inputs)
    cp=safe_load(a.checkpoint);models,e,_,guard=setup(SimpleNamespace(device='cpu'),initial_models=cp['transaction']['models'])
    e._fc1_firms=200;e.refresh_firm_target()
    for m in models.values():m.eval();m.requires_grad_(False)
    e._build_simulation_bp_grid_policy=lambda:GridBPSimulationPolicy(target_model=e._target_policy_value(),sdf_fc1_model=models['sdf_fc1'],p0_loss=e.loss_fns['p0'],pi_loss=e.loss_fns['pi'],hyperparams=e.hyperparams,economic_config=e.config,n_child_shocks=e.hyperparams.simulation_bp_grid_n_child_shocks,shock_seed=getattr(e.hyperparams,'simulation_bp_grid_shock_seed',314159),require_cuda=False)
    before=hashes(models);cfg=replace(JointCoverageConfig(),max_candidates=8,candidate_batch_size=4,bin_cap=4)
    source=dict(path=str(a.source_store),outer=12,manifest_sha256=file_hash(a.source_store/'manifest.json'))
    templates=list(committed_templates([source],outer=13))
    with torch.no_grad():
        bank,schedule,stats=build_joint_bank(safe_load(a.bank),templates,outer=13,target=4,cfg=cfg,output=a.output/'next_outer_bank',
            simulator_factory=lambda n:simulator(models,e,n,cfg.seed,device=torch.device('cpu')),
            economic=dict(config=asdict(AnalysisEconomicConfig.from_current_config()),hyperparameters=asdict(e.hyperparams)),committed=True,pool_fingerprint=source['manifest_sha256'])
        store=generate_long_horizon_dataset(models,e,bank,schedule,{'fingerprint':'FROZEN_NEXT_OUTER_POOL_CONTRACT'},a.output/'h50',outer=13,
            stage='FROZEN_NEXT_OUTER_EMPIRICAL',parent_batch_size=4,diagnostic_metadata=dict(frozen_contract_only=True,coverage_config=asdict(cfg)))
        write_dashboard(store,a.output/'diagnostics',cfg)
    verify_fixed_validation(a.inputs)
    if hashes(models)!=before or guard['calls']:raise ValueError('frozen invariant violation')
    atomic_write_json(a.output/'summary.json',dict(status='COMPLETE',kind='FROZEN_NEXT_OUTER_CONTRACT_NOT_TRAINED_OUTER13',
        source_manifest_sha256=source['manifest_sha256'],source_raw_templates=len(templates),bank_statistics=stats,
        models_unchanged=True,validation_unchanged=True,optimizer_steps=0,training_performed=False,total_seconds=time.monotonic()-started))
    print(json.dumps(stats,default=str))


if __name__=='__main__':main()
