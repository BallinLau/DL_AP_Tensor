"""Stratified CPU-only real source diagnostic. No approval and no optimizer.

Initial root failures are isolated per Parent ONLY in this diagnostic.
Production acceptance/fail-stop is unchanged. Every selected ID remains in
the scheduled denominator. Entry observation is side-effect free.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import subprocess
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.h50_source_certification import certify
from analysis.finite_direct_continuation_state import safe_load,state_digest
from analysis.nested_equilibrium_utils import atomic_write_json,restore_training_transaction
from analysis.run_fc1_self_consistent_macro import setup,select_state
from data.finite_direct_long_horizon_runtime import generate_long_horizon_dataset
from data.finite_direct_initial_hstar import InitialCalibrationBlocked
from data.finite_direct_sampling import state_hash,file_hash
from training.bp_simulation_policy import GridBPSimulationPolicy


def stratified_selection(bank, split, count, seed=20261021):
    s=bank['state'];b=s['b'].numpy();k=s['K'].numpy();alive=s['alive'].numpy().astype(bool)
    specs=bank.get('_schedule',{}).get('parents',[])
    rows=[]
    for j,pid in enumerate(s['transition_path_id'].tolist()):
        bj=b[j,alive[j]];kj=k[j,alive[j]]
        corr=float(np.corrcoef(bj,kj)[0,1]) if np.std(bj)>0 and np.std(kj)>0 else 0.
        rows.append(dict(index=j,parent_id=pid,split=split,
            sampling_source=specs[j]['sampling_source'] if specs else 'NOT_AVAILABLE',
            initial_mean_b=float(bj.mean()),initial_b_p90=float(np.quantile(bj,.9)),
            initial_K=float(kj.sum()),initial_x=float(s['x'][j]),initial_b_K_correlation=corr,
            selection_seed=seed))
    df=pd.DataFrame(rows);rng=np.random.default_rng(seed)
    features=['initial_mean_b','initial_b_p90','initial_K','initial_x','initial_b_K_correlation']
    result=[]
    for source,g in df.groupby('sampling_source',sort=True):
        n=count//2 if len(df.sampling_source.unique())==2 else count
        ids=g.index.to_numpy();v=g[features].rank(pct=True).to_numpy()
        first=int(rng.integers(len(ids)));chosen=[first];distance=((v-v[first])**2).sum(1)
        while len(chosen)<min(n,len(ids)):
            distance[chosen]=-1;z=int(np.argmax(distance));chosen.append(z)
            distance=np.minimum(distance,((v-v[z])**2).sum(1))
        result.extend(ids[chosen])
    return df.loc[result].reset_index(drop=True)


class Observer:
    def __init__(self):self.events=[];self.entry_candidates=[];self.firm_rows=[]
    def __call__(self,phase,sim,s,records,t,branch):
        byid={r['parent_candidate_id']:r for r in records}
        candidates=getattr(sim,'_pending_entry_candidates',None)
        screened=None
        if phase in ('INITIAL_PROCESSED','POST_DECISION') and t<=10 and bool(s['alive'].any()):
            from data.simulation_forward import forward_policy_value_for_simulation
            mask=s['alive'].bool();path,slot=torch.nonzero(mask,as_tuple=True)
            inputs=torch.stack([s[k][mask] for k in ('b','z','eta','i')]+[s[k][path] for k in ('x','hatcf','lnkf')],1)
            outputs=forward_policy_value_for_simulation(sim.models['policy_value'],inputs)
            values={k:getattr(outputs,k).flatten().cpu().tolist() for k in ('P0','PI','Phat','bar_z','bar_i')}
            for row,(j,c) in enumerate(zip(path.tolist(),slot.tolist())):
                self.firm_rows.append(dict(initial_parent_id=int(s['transition_path_id'][j]),t=t,branch=branch,
                    firm_id=int(s['firm_id'][j,c]),K=float(s['K'][j,c]),
                    **dict(zip(('b','z','eta','i','x','hatcf','lnkf'),inputs[row].tolist())),
                    **{k:v[row] for k,v in values.items()}))
        if phase=='POST_ENTRY_PRE_DECISION' and candidates is not None:
            from data.entry import evaluate_investment_compare_entry
            screened=evaluate_investment_compare_entry(sim.models['policy_value'],candidates,
                x=s['x'],hatcf=s['hatcf'],lnkf=s['lnkf'],chunk_size=sim.entry_inference_chunk_size)
        for j,pid in enumerate(s['transition_path_id'].tolist()):
            alive=s['alive'][j].bool();mass=s.get('firm_mass',torch.ones_like(s['K']))[j]
            hard=alive & (s['bar_z'][j]>=.5) if phase in ('INITIAL_PROCESSED','POST_DECISION') else torch.zeros_like(alive)
            entry=s['entry'][j].bool() & alive
            r=byid.get(pid,{})
            get=lambda name:float(s[name][j]) if name in s else None
            rec=dict(initial_parent_id=pid,t=t,branch=branch,phase=phase,x=get('x'),
                hatcf_pred=get('hatcf'),lnkf_pred=get('lnkf'),
                hatc_cal=r.get('Hatc_cal_t1'),lnk_cal=r.get('LnK_cal_t1'),
                K_ledger=r.get('K_cal_t1'),C_raw=r.get('C_raw_t1'),
                alive_pre_decision=int(alive.sum()),hard_default_count=int(hard.sum()),
                alive_post_exit=int((alive & ~hard).sum()),
                K_pre_exit=float((s['K'][j]*mass*alive).sum()),
                K_post_exit=float((s['K'][j]*mass*(alive & ~hard)).sum()),
                K_entry_gross=get('event_K_entry_gross') if t else None,
                K_entry_surviving=float((s['K'][j]*mass*entry*(~hard)).sum()) if t else None,
                entry_attempted=bool(t and phase!='POST_EXIT'),
                potential_entry_count=get('event_potential_count') if t and phase!='POST_EXIT' else None,
                accepted_entry_count=get('event_accepted_count') if t and phase!='POST_EXIT' else None,
                firm_mass=r.get('firm_mass'),default_mass=r.get('default_mass'),investment_mass=r.get('investment_mass'),
                economic_state_valid=r.get('economic_state_valid'),
                continuation_allowed=all(np.isfinite(r.get(v,np.nan)) for v in ('C_raw_t1','K_cal_t1','Hatc_cal_t1','LnK_cal_t1')) and r.get('C_raw_t1',0)>0 and r.get('K_cal_t1',0)>0,
                failure_reason=r.get('failure_reason'))
            self.events.append(rec)
            if phase=='POST_EXIT' and 'transition_previous_K' in s:
                previous=s['transition_previous_K'][j]
                previous_alive=s['transition_previous_alive'][j].bool()
                if previous.shape==alive.shape:
                    rec['K_exit_old']=float((previous*previous_alive*(~alive)).sum())
                    rec['delta_K_incumbent']=float(((s['K'][j]-previous)*alive).sum())
            if screened is not None:
                for c in torch.nonzero(candidates.mask[j]).flatten().tolist():
                    self.entry_candidates.append(dict(initial_parent_id=pid,t=t,branch=branch,
                        entry_P0=float(screened['entry_P0'][j,c]),entry_PI=float(screened['entry_PI'][j,c]),
                        entry_value_gap=float(screened['entry_value_gap'][j,c]),
                        accepted=bool(screened['accepted'][j,c]),candidate_z=float(candidates.z[j,c]),
                        entry_cost=float(candidates.entry_cost[j,c]),candidate_K_birth=float(candidates.K_birth[j,c])))


def frozen_cpu_fresh_state(episode, source):
    """Preserve CUDA RNG bytes; never reinterpret them as CPU generator bytes.

    Rollout uses original identity-keyed simulation CRN, not SDF training-pair
    draws. This is NOT an exact CPU resume of CUDA SDF stochastic training.
    The production A800 restore_fresh contract is deliberately unchanged.
    """
    from analysis.run_full_nested_finite_direct_outer4 import restore_fresh
    pair=source['pair_rng']
    compatible=pair is None or pair.numel()==torch.Generator(device='cpu').get_state().numel()
    restore_fresh(episode,source if compatible else dict(source,pair_rng=None))
    return dict(source_pair_rng_hash=state_digest(pair),
        source_pair_rng_bytes=0 if pair is None else pair.numel(),
        applied_to_cpu_generator=compatible and pair is not None,
        source_cuda_rng_preserved_not_converted=not compatible,
        sdf_pair_rng_used_in_frozen_rollout=False,
        exact_cuda_training_rng_resume_claimed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parents',type=int,default=32)
    a=p.parse_args()
    if a.parents not in (32,64):p.error('stratified review uses 32 or 64 Parents')
    torch.set_num_threads(4)
    def forbidden(*args,**kwargs):raise RuntimeError('optimization forbidden in frozen review')
    for c in (torch.optim.Adam,torch.optim.AdamW,torch.optim.LBFGS):c.step=forbidden
    torch.Tensor.backward=forbidden
    a.output.mkdir(parents=True,exist_ok=True)
    learning,train,train_schedule,pool,audits=certify(a.inputs/'outer_011_complete.pt',a.inputs/'source_journal_00907.pt',
        a.inputs/'latest.json',source_job=702636,artifact_dir=a.inputs)
    for k,v in audits.items():atomic_write_json(a.output/(k+'.json'),v)
    validation=safe_load(a.inputs/'validation_initial_bank.pt')
    # Original deterministic schedule identifies BETA/EMPIRICAL; retained
    # original source IDs, not a future-success-based selection.
    validation_schedule=json.loads((a.inputs/'validation_parent_sampling_schedule.json').read_text())
    if file_hash(a.inputs/'validation_initial_bank.pt')!='b7be82344320aecf75bf7c57a1580e28b2cd8282f1426bc1363fe5a8dceddab8':
        raise ValueError('fixed Validation SHA mismatch')
    if file_hash(a.inputs/'validation_parent_sampling_schedule.json')!='686a684ed35164050d1cba1dcd93eec7119b3b021445db86fed4031c3944b4b9':
        raise ValueError('fixed Validation schedule mismatch')
    models,e,fc1,guard=setup(argparse.Namespace(device='cpu'),initial_models=learning['transaction']['models'])
    from analysis.finite_direct_continuation_state import verify_optimizer_restored
    restore_training_transaction(learning['transaction'],models=models,
        optimizers={**e.optimizers,'fc1_conditional':fc1},episode=e)
    fresh_audit=frozen_cpu_fresh_state(e,learning['fresh'])
    atomic_write_json(a.output/'cpu_fresh_rng_compatibility.json',fresh_audit)
    live_restore={name:verify_optimizer_restored(learning['transaction']['optimizers'][name],opt)
                  for name,opt in {**e.optimizers,'fc1_conditional':fc1}.items()}
    atomic_write_json(a.output/'live_optimizer_restore_audit.json',live_restore)
    e._fc1_firms=200;e.refresh_firm_target()
    e._build_simulation_bp_grid_policy=lambda:GridBPSimulationPolicy(target_model=e._target_policy_value(),
        sdf_fc1_model=models['sdf_fc1'],p0_loss=e.loss_fns['p0'],pi_loss=e.loss_fns['pi'],
        hyperparams=e.hyperparams,economic_config=e.config,
        n_child_shocks=e.hyperparams.simulation_bp_grid_n_child_shocks,
        shock_seed=getattr(e.hyperparams,'simulation_bp_grid_shock_seed',314159),require_cuda=False)
    for model in models.values():model.eval();model.requires_grad_(False)
    before={k:state_digest(m.state_dict()) for k,m in models.items()}
    observer=Observer();selected=[];roots=[];rows=[];started=time.monotonic()
    for split,bank,schedule in [('train',train,train_schedule),('validation',validation,validation_schedule)]:
        selected.extend(stratified_selection(dict(bank,_schedule=schedule),split,a.parents//2).to_dict('records'))
    pd.DataFrame(selected).to_csv(a.output/'selected_parents.csv',index=False)
    for n,spec in enumerate(selected):
        split=spec['split'];bank=train if split=='train' else validation;schedule=train_schedule if split=='train' else validation_schedule
        j=int(spec['index']);pid=spec['parent_id'];s=select_state(bank['state'],slice(j,j+1),torch.device('cpu'))
        subset=dict(bank,state=s,fingerprint=state_hash(s));subschedule=dict(schedule,parents=[schedule['parents'][j]])
        try:
            with torch.inference_mode():
                store=generate_long_horizon_dataset(models,e,subset,subschedule,{'fingerprint':'CPU_STRATIFIED_SOURCE11_REVIEW_V2'},
                    a.output/'raw'/f'{split}_{pid}',outer=12,stage='FROZEN_STRATIFIED_CPU_REVIEW',
                    parent_batch_size=1,diagnostic_observer=observer)
            roots.extend(pd.read_csv(store.root/'initial_hstar_audit.csv').to_dict('records'))
            rows.extend(r for frame in store.rows() for r in frame.to_dict('records'))
        except InitialCalibrationBlocked as exc:
            roots.append(dict(parent_id=pid,split=split,root_found=False,root_status='STRICT_INITIAL_ROOT_FAILURE',audit=exc.audit))
            rows.extend(dict(split=split,parent_candidate_id=pid,t=t,child_branch=b,node_attempted=False,
                economic_state_valid=False,failure_reason='INITIAL_ROOT_FAILURE') for t in range(1,51) for b in (0,1))
        pd.DataFrame(roots).to_csv(a.output/'initial_hstar_feasibility.csv',index=False)
        pd.DataFrame(observer.events).to_csv(a.output/'economic_events.csv',index=False)
        pd.DataFrame(observer.entry_candidates).to_csv(a.output/'entry_candidates.csv',index=False)
        pd.DataFrame(observer.firm_rows).to_csv(a.output/'firm_early_decisions.csv',index=False)
        pd.DataFrame(rows).to_csv(a.output/'all_child_rows.csv',index=False)
        atomic_write_json(a.output/'progress.json',dict(completed=n+1,scheduled=len(selected),wallclock_seconds=time.monotonic()-started))
        print(f'PARENT {n+1}/{len(selected)} id={pid} complete',flush=True)
    from analysis.summarize_h50_frozen_review import CombinedStore
    from data.finite_direct_long_horizon_runtime import ShardStore
    from data.finite_direct_long_horizon_sampling import broad_firm_bank
    from data.finite_direct_runtime import conditional_child_bank
    from analysis.economic_config import AnalysisEconomicConfig
    from analysis.test_frozen_policy_root_solve import build_problem,initial_parity
    stores=[ShardStore(p) for p in sorted((a.output/'raw').glob('validation_*')) if (p/'manifest.json').exists()]
    with torch.inference_mode():
        probe=broad_firm_bank(CombinedStore(stores),split='validation',max_rows=8,seed=20261021)
        child=conditional_child_bank(models['sdf_fc1'],probe,seed=20261022,n_child_shocks=2,
            economic=AnalysisEconomicConfig.from_current_config(),device='cpu',
            m_min=e.hyperparams.pv_m_clamp_min,m_max=e.hyperparams.pv_m_clamp_max)
        problem=build_problem(e,models['policy_value'],probe.parent_states,child.children,batch_size=8,require_legacy_m1=False)
        parity=initial_parity(problem,models['policy_value'])
        if not all(r['passed'] for r in parity):raise RuntimeError('real calculated Bellman parity failure')
        atomic_write_json(a.output/'calculated_bellman_target_parity.json',dict(parity=parity,
            source_times=probe.metadata['source_time'].tolist(),kinds=probe.metadata['source_kind'],
            contract='REBUILT_CHILD_MACRO_M_AND_ORIGINAL_PRODUCTION_RHS'))
    assert before=={k:state_digest(m.state_dict()) for k,m in models.items()}
    atomic_write_json(a.output/'real_frozen_probe_manifest.json',dict(parents=len(selected),horizon=50,optimizer_steps=0,
        model_hashes=before,models_unchanged=True,fc2_calls=guard['calls'] if isinstance(guard,dict) and 'calls' in guard else 'SEE_SETUP_GUARD',
        cpu_seconds=time.monotonic()-started,
        generation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        tracked_dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()),
        source_files={name:file_hash(ROOT/name) for name in (
            'analysis/run_h50_frozen_extinction_review.py','analysis/h50_source_certification.py',
            'data/finite_direct_long_horizon_runtime.py','data/finite_direct_long_horizon_sampling.py',
            'data/finite_direct_initial_hstar.py','data/finite_direct_runtime.py',
            'analysis/run_fc1_self_consistent_macro.py','data/simulate_ts_parallel.py','models/policy_value.py')}))


if __name__=='__main__':main()
