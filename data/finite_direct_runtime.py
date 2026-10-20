"""Bounded finite-firm production adapter; no root search or optimizer.

Economic execution is delegated to the already exercised FC1 experiment's
SimulateTS adapter. Only persistence, observation and conditional-bank wiring
live here. Every cache is bound to ALL networks, CRN and operator provenance.
"""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from analysis.nested_equilibrium_utils import atomic_torch_save,atomic_write_json,stable_json_hash,tensor_hash,module_hash
from analysis.run_fc1_self_consistent_macro import simulator,process_node,select_state,hashes
from analysis.economic_config import AnalysisEconomicConfig
from analysis.convergence_transition import ConvergenceShockBank,MacroTransitionContext,build_child_exogenous_bundle
from data.simulate_ts_parallel import _expand_branches_batched
from data.finite_direct_sampling import file_hash,state_hash,validate_template_splits,SamplingConfig,fixed_schedule
from training.inner_fixed_point import InnerFPCollocationBank,InnerFPChildBank

OPERATOR='finite_direct_production_ledger_v1'


def load_locked_sampling(root):
    root=Path(root); sig=json.loads((root/'sampling_signature.json').read_text())
    for name,digest in sig['artifact_sha256'].items():
        if file_hash(root/name)!=digest: raise RuntimeError(f'sampling artifact changed: {name}')
    template=torch.load(root/'empirical_template_bank.pt',map_location='cpu',weights_only=False)
    validate_template_splits(template)
    banks={}; schedules={}
    for split in ('train','validation'):
        schedules[split]=json.loads((root/f'{split}_parent_sampling_schedule.json').read_text())
        cfg=SamplingConfig(**sig['config'])
        if schedules[split]!=fixed_schedule(cfg,template,split): raise RuntimeError('locked sampling plan changed')
        banks[split]=torch.load(root/f'{split}_initial_bank.pt',map_location='cpu',weights_only=False)
        if state_hash(banks[split]['state'])!=banks[split]['fingerprint']: raise RuntimeError('parent bank corruption')
        if banks[split]['state']['transition_path_id'].tolist()!=[r['parent_id'] for r in schedules[split]['parents']]:
            raise RuntimeError('parent identity does not match locked schedule')
        if banks[split]['schedule_fingerprint']!=schedules[split]['fingerprint']:
            raise RuntimeError('initial bank schedule fingerprint mismatch')
    if set(banks['train']['state']['transition_path_id'].tolist())&set(banks['validation']['state']['transition_path_id'].tolist()):
        raise RuntimeError('initial lineage leakage')
    return banks,schedules,sig


def cache_provenance(models,bank,schedule,signature,*,outer,stage,economic):
    result=dict(model_hashes=hashes(models),economic_hash=stable_json_hash(economic),
        operator=OPERATOR,sampling_signature=signature['fingerprint'],parent_bank_hash=bank['fingerprint'],
        crn_hash=stable_json_hash([r['child_shock_keys'] for r in schedule['parents']]),
        outer=outer,stage=stage,macro_inner=(int(stage[1:]) if stage.startswith('D') else None))
    if 'run_identity' in signature: result['run_identity']=signature['run_identity']
    return result


def verify_cache(saved,expected):
    if saved!=expected: raise RuntimeError('stale finite-direct dataset provenance')


def capture_node(state,records,*,split,outer,time,branch,model_hashes):
    """Observe existing production state after its ledger, no extra forward.

    All live incumbent/entrant input rows are kept, including nodes whose
    ledger is undefined. Eligibility is recorded separately, not inferred from
    future child success. Raw input fields are unchanged by ledger execution.
    """
    n,f=state['b'].shape; mask=state['alive'].bool()
    expand=lambda v:v[:,None].expand(n,f)
    states=torch.stack([state['b'],state['z'],state['eta'],state['i'],expand(state['x']),
                        expand(state['hatcf']),expand(state['lnkf'])],dim=-1)
    context=torch.stack([state['hatc_cal'],state['lnk_cal']],dim=-1)
    valid_by_id={r['parent_candidate_id']:r['economic_state_valid'] for r in records}
    valid=torch.tensor([valid_by_id.get(int(p),False) for p in state['transition_path_id'].tolist()],device=mask.device)
    return dict(states=states[mask].detach().cpu(),capital=state['K'][mask].detach().cpu(),
        context=context[:,None,:].expand(n,f,2)[mask].detach().cpu(),
        path=expand(state['transition_path_id'])[mask].detach().cpu(),
        firm_id=state['firm_id'][mask].detach().cpu(),eligible=expand(valid)[mask].detach().cpu(),
        split=split,source_outer=outer,source_time=time,source_branch=branch,model_hashes=model_hashes,
        source_kind='ENDOGENOUS_SIMULATED_PARENT_DISTRIBUTION' if time>=1 else 'RECORDED_INITIAL_PARENT_DISTRIBUTION',
        state_columns=['b','z','eta','i','x','hatcf','lnkf'],
        timing='post-entry incumbent-decision inputs at node; ledger context from same node; no future-outcome selection')


@torch.no_grad()
def generate_dataset(models,episode,bank,schedule,signature,output,*,outer,stage,parent_batch_size=128,capture=True):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    economic=asdict(AnalysisEconomicConfig.from_current_config())
    economic['hyperparameters']=asdict(episode.hyperparams)
    economic['simulation_kwargs']=dict(entry_capital_ratio=episode.hyperparams.entry_capital_ratio,
        entry_mode=episode.hyperparams.entry_mode,entry_size_ratio=episode.hyperparams.entry_size_ratio,
        entry_cost_max=episode.hyperparams.entry_cost_max,
        aggregation='FINITE_FIRM_DIRECT_AGGREGATION',
        bp_source='grid',raw_consumption=True)
    provenance=cache_provenance(models,bank,schedule,signature,outer=outer,stage=stage,economic=economic)
    provenance['capture']=capture
    p=output/'provenance.json'
    if p.exists(): verify_cache(json.loads(p.read_text()),provenance)
    else: atomic_write_json(p,provenance)
    split=schedule['split']; seed=schedule['seed']; initial=hashes(models); shards=[]; captures=[]
    state0=bank['state']; n=len(state0['x'])
    for start in range(0,n,parent_batch_size):
        end=min(start+parent_batch_size,n); target=output/f'batch_{start:06d}_{end:06d}.pt'
        digest=target.with_suffix('.sha256.json')
        if target.exists() and digest.exists():
            if file_hash(target)!=json.loads(digest.read_text())['sha256']: raise RuntimeError('corrupt simulation shard')
            chunk=torch.load(target,map_location='cpu',weights_only=False)
            verify_cache(chunk['provenance'],provenance)
        else:
            s=select_state(state0,slice(start,end),episode.device)
            sim=simulator(models,episode,end-start,seed); snaps=[]
            s,parents,valid=process_node(sim,s,0,-1,entry=False)
            if capture: snaps.append(capture_node(s,parents,split=split,outer=outer,time=0,branch=-1,model_hashes=initial))
            rows=[]; pmap={r['parent_candidate_id']:r for r in parents}
            for r in parents:
                if not r['economic_state_valid']:
                    for b in (0,1): rows.append(dict(parent_candidate_id=r['parent_candidate_id'],t=1,child_branch=b,
                        split=split,economic_state_valid=False,failure_reason='PARENT_'+r['failure_reason']))
            if valid:
                s=select_state(s,torch.tensor(valid,device=episode.device))
                children=_expand_branches_batched(sim,s,child_t=1)
                for branch,child in enumerate(children):
                    child,observations,_=process_node(sim,child,1,branch,entry=True)
                    if capture: snaps.append(capture_node(child,observations,split=split,outer=outer,time=1,branch=branch,model_hashes=initial))
                    for r in observations:
                        par=pmap[r['parent_candidate_id']]
                        r.update(split=split,x_t=par['x_t1'],Hatc_cal_t=par['Hatc_cal_t1'],LnK_cal_t=par['LnK_cal_t1'],
                                 Hatcf_t=par['Hatcf_t1'],LnKF_t=par['LnKF_t1'])
                        rows.append(r)
                del children
            chunk=dict(provenance=provenance,rows=rows,captures=snaps)
            atomic_torch_save(target,chunk); atomic_write_json(digest,dict(sha256=file_hash(target)))
            del s,sim
        shards.extend(chunk['rows']); captures.extend(chunk['captures']); del chunk
    if hashes(models)!=initial: raise RuntimeError('simulation changed frozen network')
    frame=pd.DataFrame(shards)
    # Undefined parent-only shards may not have these columns: keep NaN audits,
    # never impute an eligible economic target.
    for name in ['x_t','x_t1','Hatc_cal_t','LnK_cal_t','Hatc_cal_t1','LnK_cal_t1','C_raw_t1','K_cal_t1']:
        if name not in frame: frame[name]=np.nan
    if frame.duplicated(['split','parent_candidate_id','t','child_branch']).any(): raise RuntimeError('duplicate child')
    if len(frame)!=2*n: raise RuntimeError('missing attempted parent/two-child observations')
    frame.to_csv(output/'observations.csv',index=False)
    manifest=dict(provenance=provenance,rows=len(frame),observations_sha256=file_hash(output/'observations.csv'),
                  captures=len(captures),shards={p.name:file_hash(p) for p in sorted(output.glob('batch_*.pt'))})
    atomic_write_json(output/'manifest.json',manifest)
    return frame,captures,manifest


def firm_bank_from_captures(captures,*,split,max_rows,seed):
    selected=[s for s in captures if s['split']==split and s['source_time']==0]
    if not selected: raise RuntimeError('no finite-direct parent snapshots')
    states=torch.cat([s['states'][s['eligible']] for s in selected]); context=torch.cat([s['context'][s['eligible']] for s in selected])
    paths=torch.cat([s['path'][s['eligible']] for s in selected]); ids=torch.cat([s['firm_id'][s['eligible']] for s in selected])
    if not len(states): raise RuntimeError('no legal conditional Firm parents')
    g=torch.Generator().manual_seed(seed); idx=torch.randperm(len(states),generator=g)[:max_rows]
    source=dict(path=paths[idx],firm_id=ids[idx],true_macro_context=context[idx],
                source_network_hashes=selected[0]['model_hashes'],operator=OPERATOR)
    return InnerFPCollocationBank(states[idx],torch.zeros(len(idx),dtype=torch.long),seed,split,source)


def check_firm_lineage(train,validation):
    if set(train.metadata['path'].tolist())&set(validation.metadata['path'].tolist()): raise RuntimeError('Firm lineage leakage')


@torch.no_grad()
def conditional_child_bank(model,bank,*,seed,n_child_shocks,economic,device,chunk_size=512,m_min=.7,m_max=1.3):
    """Use SAME observed 7D row + SAME-node true macro context; exact eta later.

    Generic build_collocation_bank mixes marginal draws and build_child_bank
    defaults to perceived parent macro. Neither is valid for this adapter.
    We reuse their actual exogenous transition primitive with explicit context.
    """
    before=module_hash(model); n=len(bank.parent_states)
    shock=ConvergenceShockBank.create(n,n_child_shocks,seed=seed)
    out=[]; raw=[]
    for start in range(0,n,chunk_size):
        stop=min(start+chunk_size,n); p=bank.parent_states[start:stop].to(device)
        context=bank.metadata['true_macro_context'][start:stop].to(device)
        sub=shock.gather(torch.arange(start,stop))
        b=build_child_exogenous_bundle(model,p,MacroTransitionContext(context[:,0:1],context[:,1:2]),sub,economic_config=economic)
        child=torch.stack([p[:,0,None].expand(-1,n_child_shocks),b.z_next[...,0],
            torch.zeros_like(b.eta_next[...,0]),b.i_next[...,0],b.x_next[...,0],
            b.hatcf_next[...,0],b.lnkf_next[...,0],b.m_raw[...,0].clamp(m_min,m_max)],dim=-1)
        out.append(child.cpu()); raw.append(b.m_raw.cpu())
    if module_hash(model)!=before: raise RuntimeError('Macro changed during conditional children')
    return InnerFPChildBank(torch.cat(out),shock.eps_x,shock.eps_z,shock.u_eta,shock.u_i,seed,
        dict(sdf_fc1_hash=before,parent_fingerprint=bank.fingerprint,raw_m=torch.cat(raw),
             common_m_clamp=[m_min,m_max],context='same_node_true_macro',eta='integrated_exactly_in_operator'))
