"""Small frozen real-checkpoint feasibility gate, never an optimizer step.

CPU Grid's existing reference option is explicitly enabled ONLY in this probe.
Do not interpret its wallclock as A800 performance.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import pandas as pd

from analysis.finite_direct_continuation_state import safe_load, state_digest
from analysis.nested_equilibrium_utils import atomic_write_json
from analysis.run_fc1_self_consistent_macro import setup, simulator, select_state
from data.finite_direct_initial_hstar import calibrate_initial, InitialCalibrationBlocked
from data.finite_direct_sampling import file_hash
from data.market_clearing import MarketClearingConfig
from training.bp_simulation_policy import GridBPSimulationPolicy


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--parents', type=int, default=4)
    p.add_argument('--rollout', action='store_true')
    p.add_argument('--bellman-probe', action='store_true')
    a = p.parse_args(argv)
    if not 1 <= a.parents <= 10:
        p.error('forward-only probe limited to 1..10 real parents')
    torch.set_num_threads(4)
    def forbidden(*args, **kwargs):
        raise RuntimeError('optimizer step forbidden in feasibility probe')
    for cls in (torch.optim.Adam, torch.optim.AdamW, torch.optim.LBFGS):
        cls.step = forbidden
    checkpoint = safe_load(a.checkpoint)
    transaction = checkpoint['transaction']
    models, e, _, guard = setup(argparse.Namespace(device='cpu'),
                                initial_models=transaction['models'])
    e._fc1_firms = 200
    e.refresh_firm_target()
    e._build_simulation_bp_grid_policy = lambda: GridBPSimulationPolicy(
        target_model=e._target_policy_value(), sdf_fc1_model=models['sdf_fc1'],
        p0_loss=e.loss_fns['p0'], pi_loss=e.loss_fns['pi'], hyperparams=e.hyperparams,
        economic_config=e.config, n_child_shocks=e.hyperparams.simulation_bp_grid_n_child_shocks,
        shock_seed=getattr(e.hyperparams,'simulation_bp_grid_shock_seed',314159), require_cuda=False)
    bank = safe_load(a.bank)
    state = select_state(bank['state'], slice(0, a.parents), torch.device('cpu'))
    seed=bank['seed']
    sim = simulator(models, e, a.parents, seed, device=torch.device('cpu'))
    a.output.mkdir(parents=True, exist_ok=True)
    initial = {k: state_digest(v.state_dict()) for k, v in models.items()}
    started = time.monotonic()
    result = dict(checkpoint_sha256=file_hash(a.checkpoint), bank_sha256=file_hash(a.bank),
        source_stage='outer_011/outer_commit (verified remote Job702636)', parents=a.parents, device='CPU',
        real_root_feasibility='NOT_MEASURED', optimizer_steps=0,
        synthetic_policy=False, cpu_grid_reference=True,
        root_config=asdict(MarketClearingConfig()), model_hashes=initial)
    result['generation_source']=dict(
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
        tracked_dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()),
        code_sha256={name:file_hash(ROOT/name) for name in (
            'data/finite_direct_initial_hstar.py','data/finite_direct_long_horizon_runtime.py',
            'data/finite_direct_long_horizon_sampling.py','data/market_clearing.py',
            'data/simulate_ts_parallel.py','analysis/check_initial_hstar_cpu.py')})
    try:
        _, _, _, audit = calibrate_initial(sim, state, outer=12, split='validation',
            economic=asdict(e.hyperparams), crn={'seed':seed},
            cache=a.output/'initial_cache.pt')
        result.update(real_root_feasibility='PASS', audit=audit)
        pd.DataFrame(audit).to_csv(a.output/'initial_hstar_audit.csv', index=False)
        if a.rollout:
            from data.finite_direct_long_horizon_runtime import generate_long_horizon_dataset
            subset=dict(bank,state=state)
            from data.finite_direct_sampling import state_hash
            subset['fingerprint']=state_hash(state)
            schedule=dict(split='validation',seed=seed,parents=[dict(child_shock_keys=[
                dict(branch=b,path_id=int(pid)) for b in (0,1)]) for pid in state['transition_path_id']])
            store=generate_long_horizon_dataset(models,e,subset,schedule,
                {'fingerprint':'CPU_REVIEW_REAL_VALIDATION_SUBSET'},a.output/'rollout50',
                outer=12,stage='FROZEN_CPU_FORWARD_ONLY',parent_batch_size=a.parents)
            result['rollout_manifest']=store.manifest
            if a.bellman_probe:
                from data.finite_direct_long_horizon_sampling import broad_firm_bank
                from data.finite_direct_runtime import conditional_child_bank
                from analysis.economic_config import AnalysisEconomicConfig
                from analysis.test_frozen_policy_root_solve import build_problem,evaluate_problem,initial_parity,residual_tensors
                probe=broad_firm_bank(store,split='validation',max_rows=8,seed=20261021)
                child=conditional_child_bank(models['sdf_fc1'],probe,seed=20261022,
                    n_child_shocks=2,economic=AnalysisEconomicConfig.from_current_config(),device='cpu',
                    m_min=e.hyperparams.pv_m_clamp_min,m_max=e.hyperparams.pv_m_clamp_max)
                problem=build_problem(e,models['policy_value'],probe.parent_states,child.children,
                    batch_size=8,require_legacy_m1=False)
                metrics=evaluate_problem(models['policy_value'],problem)
                parity=initial_parity(problem,models['policy_value'])
                if not all(r['passed'] for r in parity):
                    raise RuntimeError('calculated/on-policy production Bellman RHS parity failed')
                with torch.no_grad():values=residual_tensors(models['policy_value'],problem[0])
                points=pd.DataFrame(probe.parent_states.numpy(),columns=['b','z','eta','i','x','hatcf','lnkf'])
                points['p0_normalized']=values['p0_normalized'].reshape(-1).cpu().numpy()
                points['pi_normalized']=values['pi_normalized'].reshape(-1).cpu().numpy()
                points['source_kind']=probe.metadata['source_kind']
                points.to_csv(a.output/'bellman_probe_rows.csv',index=False)
                from analysis.h50_diagnostics import write_bellman_heatmap
                write_bellman_heatmap(points,a.output/'firm_bellman_residual_macro_heatmap.png')
                result['bellman_forward_probe']=dict(rows=8,calculated_rows=4,
                    parent_fingerprint=probe.fingerprint,child_provenance=child.metadata['parent_fingerprint'],
                    source_times=probe.metadata['source_time'].tolist(),metrics=metrics,target_parity=parity,
                    label='SMALL_STRATIFIED_FORWARD_ONLY; NOT_POPULATION_CONVERGENCE')
    except InitialCalibrationBlocked as exc:
        result.update(real_root_feasibility='FAIL', blocker=str(exc), audit=exc.audit)
        rows = exc.audit.get('failures', []) if isinstance(exc.audit, dict) else exc.audit
        pd.DataFrame(rows).to_csv(a.output/'initial_hstar_audit.csv', index=False)
    result.update(wallclock_seconds=time.monotonic()-started,
        models_unchanged=initial == {k:state_digest(v.state_dict()) for k,v in models.items()},
        fc2_forward_calls=guard['calls'], gpu_performance='NOT_MEASURED',
        training_performed=False, slurm_submitted=False)
    atomic_write_json(a.output/'root_feasibility.json', result)
    print({k:v for k,v in result.items() if k not in ('audit','model_hashes','root_config')}, flush=True)
    return 0 if result['real_root_feasibility']=='PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
