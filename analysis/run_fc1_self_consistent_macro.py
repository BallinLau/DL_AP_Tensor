#!/usr/bin/env python3
"""Eight conditional FC1 updates using finite-firm production aggregation.

Economic transitions, BP grid actions and entry use SimulateTS primitives.
Undefined economic rows are retained, never repaired or used as labels.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import Config
from experiments.run_utils import build_models, build_optimizers, build_hyperparams
from training.episode import Episode, SDFTrainingPhase
from data.simulate_ts import SimulateTS
from data.simulate_ts_parallel import (
    _initialize_batched_state, _process_node_batched, _expand_branches_batched,
    _apply_entry_batched, _apply_exit_batched, _keyed_normal,
)
from data.market_clearing import MarketClearingController
from data.boundary_aware_market_clearing import BoundaryAwareMarketClearingController
from analysis.macro_parent_state_bank import _simulation_kwargs
from analysis.run_macro_outer_inner_pq import module_hash, install_fc2_guard


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str, allow_nan=True) + '\n')
    temp.replace(path)


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp); temp.replace(path)


def digest_state(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        if isinstance(value, torch.Tensor):
            v = value.detach().cpu().contiguous()
            h.update(name.encode()); h.update(str(v.dtype).encode())
            h.update(str(tuple(v.shape)).encode()); h.update(v.numpy().tobytes())
    return h.hexdigest()


def hashes(models):
    m = models['sdf_fc1']
    return dict(firm=module_hash(models['policy_value']), sdf=module_hash(m.sdf_model),
                wealth=module_hash(m.value_model), fc1_c=module_hash(m.fc1_model.FC1_C),
                fc1_k=module_hash(m.fc1_model.FC1_K))


def frozen_check(initial, current):
    for key in ('firm', 'sdf', 'wealth'):
        if initial[key] != current[key]:
            raise RuntimeError(f'frozen parameter hash changed: {key}')


def select_state(state, index, device=None):
    size = len(state['x'])
    return {k: (v[index].detach().clone().to(device or v.device)
                if isinstance(v, torch.Tensor) and v.ndim and len(v) == size
                else deepcopy(v)) for k, v in state.items()}


def setup(args):
    device = torch.device(args.device)
    Config.DEVICE = device
    hp = build_hyperparams()
    hp.entry_mode = 'investment_compare'; hp.entry_spec_version = 'investment_compare_v1'
    hp.entry_capital_ratio = .05; hp.entry_size_ratio = 1.; hp.entry_cost_max = .5
    hp.entry_dummy_i = 0.; hp.consumption_aggregation_mode = 'raw'; hp.consumption_floor = 0.
    hp.node_accounting_mode = 'economic_node_ledger'
    hp.aggregation_mode = 'legacy_mixed_replay'
    hp.simulation_bp_action_source = 'grid'; hp.simulation_bp_grid_n_child_shocks = 2
    hp.market_clearing_mode = 'disabled'
    hp.fc1_recursive_aux_training_enabled = False
    hp.fc1_rollout_diagnostic_enabled = False
    models = build_models(device)
    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    for name in ('policy_value', 'sdf_fc1'):
        models[name].load_state_dict(cp['models'][name], strict=True)
    for model in models.values(): model.requires_grad_(False); model.eval()
    guard = {'calls': 0}
    install_fc2_guard(models.get('fc2'), guard)
    opts = build_optimizers(models, hp)
    episode = Episode(models=models, optimizers=opts, config=Config, hyperparams=hp,
                      device=device, episode_id=1, q_checkpoint_loaded=False)
    episode.add_FC1loss = True; episode.set_sdf_training_phase(SDFTrainingPhase.FC1_ONLY)
    # This optimizer contains exactly the two FC1 heads; economic weights and
    # the one-step loss are production Episode._compute_fc1_only_loss.
    optimizer = torch.optim.AdamW(models['sdf_fc1'].fc1_model.parameters(),
                                 lr=hp.fc1_lr, weight_decay=hp.fc1_weight_decay)
    return models, episode, optimizer, guard


def simulator(models, episode, n, seed, *, device=None):
    kwargs = _simulation_kwargs(episode, seed + 11, seed + 29)
    kwargs.update(market_clearing_mode='disabled', aggregation_mode='legacy_mixed_replay')
    sim = SimulateTS(models={k: models[k] for k in ('policy_value','sdf_fc1')},
                     config=Config, n_paths=n, group_size=episode._fc1_firms,
                     horizon=1, branch_num=2, device=device or episode.device, **kwargs)
    sim.diagnostic_retain_undefined_nodes = True
    grid = sim.bp_grid_policy
    def defined_grid(**context):
        # Undefined macro accounts have no economic continuation action. They
        # keep NaN BP, never a head fallback. Valid nodes execute the original
        # production resolver with exactly the original context.
        h,l=context['hatc_cal'],context['lnk_cal']
        mask=(torch.isfinite(h)&torch.isfinite(l)).reshape(-1)
        result=torch.full_like(context['bp_head'],float('nan'))
        if bool(mask.all()): return grid(**context)
        if bool(mask.any()):
            sub={name:(value[mask] if isinstance(value,torch.Tensor) and value.ndim and len(value)==len(mask) else value)
                 for name,value in context.items()}
            result[mask]=grid(**sub)
        return result
    sim.bp_grid_policy=defined_grid
    if sim.market_clearing_controller is not None: raise RuntimeError('clearing unexpectedly enabled')
    return sim


def make_bank(models, episode, args, n, split):
    seed = args.seed + (100000 if split == 'validation' else 0)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    sim = simulator(models, episode, n, seed, device=torch.device('cpu'))
    state = _initialize_batched_state(sim, args.firms)
    # Stratified parameters; micro draws are distinct and stationary-law legal.
    u = torch.quasirandom.SobolEngine(6, scramble=True, seed=seed).draw(n).numpy()
    alpha = 1.8 + 1.4*u[:,0]; beta = 4.8 + 3.*u[:,1]
    stats = []
    for j in range(n):
        rng = np.random.default_rng(seed + 97*j)
        b = rng.beta(alpha[j], beta[j], size=args.firms)
        power = .5 + 1.5*u[j,3]
        weights = rng.random(args.firms)**power; weights /= weights.sum()
        # Vary b-k dependence by rank assignment without changing marginals.
        if u[j,4] < 1/3: b[np.argsort(weights)] = np.sort(b)
        elif u[j,4] > 2/3: b[np.argsort(weights)] = np.sort(b)[::-1]
        k = (45. + 40.*u[j,2])*weights
        state['b'][j] = torch.tensor(b, dtype=torch.float32)
        state['K'][j] = torch.tensor(k, dtype=torch.float32)
        z = state['z'][j].numpy()
        stats.append(dict(parent_id=j+(1000000 if split=='validation' else 0), split=split,
                          alpha=float(alpha[j]), beta=float(beta[j]), mean_b=float(b.mean()),
                          std_b=float(b.std()), b_p90=float(np.quantile(b,.9)),
                          b_tail_gt_half=float((b>.5).mean()), z_mean=float(z.mean()),
                          z_std=float(z.std()), z_skew=float(((z-z.mean())**3).mean()/max(z.std()**3,1e-15)),
                          K_total=float(k.sum()), K_hhi=float((weights**2).sum()),
                          b_k_corr=float(np.corrcoef(b,k)[0,1]), x=float(state['x'][j])))
    state['lnkf'] = torch.log(state['K'].sum(1))
    state['K_birth'] = state['K'].clone(); state['bp'] = state['b'].clone()
    state['transition_path_id'] += (1000000 if split=='validation' else 0)
    return state, pd.DataFrame(stats), seed


def install_guards(counters):
    def forbidden(*a, **kw):
        counters['ROOT_SEARCH_CALLS'] += 1
        raise RuntimeError('root / FAST clearing forbidden in this experiment')
    MarketClearingController.clear_state = forbidden
    BoundaryAwareMarketClearingController.clear_state = forbidden


def reason(c, k, mass, h, l):
    if mass <= 0: return 'ZERO_SURVIVING_MASS'
    if not math.isfinite(k) or k <= 0: return 'UNDEFINED_LNK'
    if not math.isfinite(c) or c <= 0: return 'NONPOSITIVE_CONSUMPTION'
    if not math.isfinite(h) or not math.isfinite(l): return 'OTHER_ECONOMIC_FAILURE'
    return ''


def process_node(sim, state, t, branch, *, entry):
    # Empty paths are handled by the caller before entering production.
    if entry: state = _apply_entry_batched(sim, state, max(20, int(sim.group_size*.1)))
    firm, macro = _process_node_batched(sim, state, t, branch)
    cols = {name:j for j,name in enumerate(sim.MACRO_COLUMNS)}
    fcols = {name:j for j,name in enumerate(sim.FIRM_COLUMNS)}
    n = len(state['x']); survival=torch.zeros(n,device=sim.device)
    invest=torch.zeros_like(survival); default=torch.zeros_like(survival)
    if len(firm):
        idx=firm[:,fcols['path']].long(); mass=firm[:,fcols['firm_mass']]
        survival.index_add_(0,idx,mass*(firm[:,fcols['Bar_z']]<.5))
        default.index_add_(0,idx,mass*firm[:,fcols['Bar_z']])
        invest.index_add_(0,idx,mass*firm[:,fcols['Bar_i']])
    metrics = torch.stack((survival,default,invest),dim=1).cpu().numpy()
    values=macro.detach().cpu().numpy()
    mapping={int(r[cols['path']]):r for r in values}
    records=[]; valid=[]
    for j,pid in enumerate(state['transition_path_id'].cpu().tolist()):
        r=mapping.get(j)
        if r is None:
            records.append(dict(parent_candidate_id=int(pid),t=t,child_branch=branch,
                                economic_state_valid=False,failure_reason='ZERO_SURVIVING_MASS'))
            continue
        get=lambda name: float(r[cols[name]])
        C,K=get('C_raw'),get('K'); hf,lf=get('hatcf'),get('lnkf')
        hcal=math.log(C/K) if C>0 and K>0 else float('nan')
        lcal=math.log(K) if K>0 else float('nan')
        fail=reason(C,K,float(metrics[j,0]),hcal,lcal)
        G=math.exp(hf)*K-C if math.isfinite(hf) and hf<700 else float('nan')
        rec=dict(parent_candidate_id=int(pid),initial_distribution_id=int(pid),t=t,
                 child_branch=branch,x_t1=get('x'),Hatcf_t1=hf,LnKF_t1=lf,
                 Hatc_cal_t1=hcal,LnK_cal_t1=lcal,C_raw_t1=C,K_cal_t1=K,
                 G_at_prediction=G,normalized_G_at_prediction=abs(G)/max(1.,abs(C),abs(K)),
                 firm_mass=get('firm_mass'),surviving_mass=float(metrics[j,0]),
                 default_mass=float(metrics[j,1]),investment_mass=float(metrics[j,2]),
                 entry_mass=get('accepted_count'),economic_state_valid=not bool(fail),failure_reason=fail,
                 I_oper=get('I_oper'),I_entry=get('I_entry'),I_maintenance=get('I_maintenance'))
        if not fail:
            # Exact transformations for the experiment's specified FC1 inputs.
            state['hatc_cal'][j]=hcal; state['lnk_cal'][j]=lcal
            valid.append(j)
        records.append(rec)
    del firm,macro
    return state,records,valid


@torch.no_grad()
def simulate_bank(models,episode,args,bank,seed,split,path,*,horizon=1):
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    manifest=dict(fc1_hash=module_hash(models['sdf_fc1'].fc1_model),bank_hash=digest_state(bank),
                  seed=seed,split=split,horizon=horizon,firms=args.firms,
                  simulation_representation='FINITE_FIRM_DIRECT_AGGREGATION')
    manifest_path=path/'simulation_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise RuntimeError('refusing cached simulation from different FC1, bank or CRN')
    atomic_json(manifest_path,manifest)
    started=time.perf_counter(); all_rows=[]
    for start in range(0,len(bank['x']),args.parent_batch_size):
        end=min(start+args.parent_batch_size,len(bank['x']))
        shard=path/f'batch_{start:06d}_{end:06d}.csv'
        if shard.exists():
            all_rows.extend(pd.read_csv(shard).to_dict('records')); continue
        state=select_state(bank,slice(start,end),episode.device)
        sim=simulator(models,episode,len(state['x']),seed)
        chunk=[]
        for t in range(horizon):
            # Production caches an already-processed main child when promoted.
            if t==0:
                state,parents,valid=process_node(sim,state,0,-1,entry=False)
            else:
                state,parents,valid=process_node(sim,state,t,-1,entry=False)
            bad=[r for r in parents if not r['economic_state_valid']]
            for rec in bad:
                for branch in range(2):
                    chunk.append(dict(parent_candidate_id=rec['parent_candidate_id'],
                        initial_distribution_id=rec['parent_candidate_id'],child_branch=branch,t=t+1,
                        economic_state_valid=False,failure_reason='PARENT_'+rec['failure_reason']))
            if not valid: break
            state=select_state(state,torch.tensor(valid,device=episode.device))
            parent_map={r['parent_candidate_id']:r for r in parents if r['economic_state_valid']}
            branches=_expand_branches_batched(sim,state,child_t=t+1)
            next_state=None
            for branch,child in enumerate(branches):
                child,rows,child_valid=process_node(sim,child,t+1,branch,entry=True)
                for rec in rows:
                    par=parent_map[rec['parent_candidate_id']]
                    rec.update(x_t=par['x_t1'],Hatc_cal_t=par['Hatc_cal_t1'],LnK_cal_t=par['LnK_cal_t1'])
                    rec['split']=split
                    chunk.append(rec)
                if branch==0 and child_valid:
                    child=_apply_exit_batched(child)
                    next_state=select_state(child,torch.tensor(child_valid,device=episode.device))
            del branches
            if next_state is None: break
            state=next_state
        # Preserve the original dynamic schedule, including undefined suffixes.
        seen={(r['parent_candidate_id'],r['child_branch'],r['t']) for r in chunk}
        for pid in bank['transition_path_id'][start:end].tolist():
            for step in range(1,horizon+1):
                for branch in range(2):
                    if (int(pid),branch,step) not in seen:
                        chunk.append(dict(parent_candidate_id=int(pid),initial_distribution_id=int(pid),
                            child_branch=branch,t=step,economic_state_valid=False,
                            failure_reason='ANCESTOR_UNDEFINED_OR_EXTINCT'))
        for r in chunk: r.setdefault('split',split)
        frame=pd.DataFrame(chunk); frame.to_csv(shard,index=False)
        all_rows.extend(chunk)
        del state,sim
        print(json.dumps(dict(stage='simulation_batch',split=split,start=start,end=end,
                              rows=len(chunk),seconds=time.perf_counter()-started)),flush=True)
    frame=pd.DataFrame(all_rows)
    if frame.duplicated(['parent_candidate_id','child_branch','t']).any():
        raise RuntimeError('duplicate economic observation')
    frame.to_csv(path/'observations.csv',index=False)
    atomic_json(path/'metadata.json',dict(seconds=time.perf_counter()-started,
                fingerprint=hashlib.sha256((path/'observations.csv').read_bytes()).hexdigest(),
                initial_bank_fingerprint=digest_state(bank),seed=seed,horizon=horizon,
                requested_parents=len(bank['x']),representation='FINITE_FIRM_DIRECT_AGGREGATION'))
    return frame,time.perf_counter()-started


def valid_frame(frame):
    return frame.loc[frame.economic_state_valid.fillna(False).astype(bool)].copy()


def xy(frame,device):
    f=valid_frame(frame)
    X=torch.tensor(f[['x_t','x_t1','Hatc_cal_t','LnK_cal_t']].to_numpy(),dtype=torch.float32,device=device)
    Y=torch.tensor(f[['Hatc_cal_t1','LnK_cal_t1']].to_numpy(),dtype=torch.float32,device=device)
    if not torch.isfinite(X).all() or not torch.isfinite(Y).all(): raise RuntimeError('nonfinite legal label')
    return X,Y,f


@torch.no_grad()
def predictions(model,X):
    c,k=model.fc1_model(X,return_physical=True)
    out=torch.cat((c,k),dim=-1)
    if not torch.isfinite(out).all(): raise RuntimeError('nonfinite FC1 output')
    return out


def regress(y,p):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float); e=p-y
    s=float(((y-y.mean())**2).sum())
    return dict(r2=1-float((e**2).sum())/s if s>0 else float('nan'),
                rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.abs(e).mean()),bias=float(e.mean()))


def evaluate(model,frame,device):
    X,Y,_=xy(frame,device)
    if not len(X): return {}
    p=predictions(model,X).cpu().numpy(); y=Y.cpu().numpy(); x=X.cpu().numpy()
    result={}
    for j,name in enumerate(('hatc','lnk')):
        result.update({name+'_'+k:v for k,v in regress(y[:,j],p[:,j]).items()})
    result.update({'delta_lnk_'+k:v for k,v in regress(y[:,1]-x[:,3],p[:,1]-x[:,3]).items()})
    return result


def production_loss(episode,X,Y):
    parent=torch.zeros((len(X),9),device=X.device)
    child=torch.zeros((len(X),1,10),device=X.device)
    parent[:,4]=X[:,0]; parent[:,5:7]=X[:,2:4]; parent[:,7:9]=X[:,2:4]
    child[:,0,4]=X[:,1]; child[:,0,8:10]=Y
    return episode._compute_fc1_only_loss({},parent,child)


def save_checkpoint(path,models,optimizer,iteration,epoch,extra):
    atomic_torch(path,dict(models={n:m.state_dict() for n,m in models.items() if n in ('policy_value','sdf_fc1')},
                 optimizer=optimizer.state_dict(),scheduler=None,iteration=iteration,epoch=epoch,
                 hashes=hashes(models),rng={'python':random.getstate(),'numpy':np.random.get_state(),
                 'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
                 **extra))


def restore_rng(rng):
    random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'].cpu())
    if rng['cuda']: torch.cuda.set_rng_state_all([v.cpu() for v in rng['cuda']])


def train_iteration(models,episode,optimizer,args,train,val,iteration,out,initial):
    out.mkdir(parents=True,exist_ok=True); model=models['sdf_fc1']
    X,Y,frame=xy(train,episode.device); V,W,vframe=xy(val,episode.device)
    if not len(X) or not len(V): raise RuntimeError('NO_LEGAL_FC1_LABELS')
    for m in models.values(): m.requires_grad_(False); m.eval()
    model.fc1_model.requires_grad_(True)
    before=hashes(models); rows=[]; best=float('inf'); best_epoch=0; best_state=None; steps=0; start_epoch=1
    latest=out/'epoch_latest.pt'
    if latest.exists():
        cp=torch.load(latest,map_location=episode.device,weights_only=False)
        model.load_state_dict(cp['models']['sdf_fc1']); optimizer.load_state_dict(cp['optimizer'])
        restore_rng(cp['rng']); start_epoch=cp['epoch']+1
        rows=cp['history']; steps=cp['optimizer_steps']; best=cp['best_loss']; best_epoch=cp['best_epoch']; best_state=cp['best_state']
        before=cp['hashes_before']
    started=time.perf_counter()
    parent_ids=torch.tensor(frame.parent_candidate_id.to_numpy(),device=episode.device)
    unique=torch.unique(parent_ids)
    for epoch in range(start_epoch,args.epochs+1):
        order=unique[torch.randperm(len(unique),device=episode.device)]
        visited=[]; weighted_loss=0.; count=0
        for begin in range(0,len(order),args.fit_parent_batch_size):
            ids=order[begin:begin+args.fit_parent_batch_size]
            mask=torch.isin(parent_ids,ids); xb,yb=X[mask],Y[mask]
            optimizer.zero_grad(set_to_none=True)
            loss=production_loss(episode,xb,yb)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite FC1 loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.fc1_model.parameters()):
                raise RuntimeError('nonfinite FC1 gradient')
            optimizer.step(); steps+=1
            if any(not torch.isfinite(p).all() for p in model.fc1_model.parameters()): raise RuntimeError('nonfinite FC1 parameter')
            weighted_loss+=float(loss)*len(xb); count+=len(xb); visited.extend(ids.cpu().tolist())
        if count!=len(frame) or len(set(visited))!=len(unique): raise RuntimeError('incomplete epoch coverage')
        with torch.no_grad(): vl=float(production_loss(episode,V,W))
        if vl<best:
            best=vl; best_epoch=epoch; best_state=deepcopy(model.fc1_model.state_dict())
        row=dict(iteration=iteration,epoch=epoch,optimizer_steps=steps,
                 train_loss=weighted_loss/count,validation_loss=vl,
                 unique_parents_visited=len(unique),unique_children_visited=count)
        row.update({'train_'+k:v for k,v in evaluate(model,train,episode.device).items()})
        row.update({'validation_'+k:v for k,v in evaluate(model,val,episode.device).items()})
        rows.append(row)
        frozen_check(initial,hashes(models))
        save_checkpoint(latest,models,optimizer,iteration,epoch,
            dict(history=rows,optimizer_steps=steps,best_loss=best,best_epoch=best_epoch,
                 best_state=best_state,hashes_before=before))
        pd.DataFrame(rows).to_csv(out/'history.csv',index=False)
        print(json.dumps(dict(stage='fc1_epoch',**row)),flush=True)
    # Keep the last epoch model and its exactly matched optimizer. Best epoch
    # is recorded, not used to reset optimization state between iterations.
    atomic_torch(out/'best_fc1.pt',best_state)
    summary=dict(iteration=iteration,epochs_completed=len(rows),optimizer_steps=steps,
                 best_epoch=best_epoch,best_validation_loss=best,seconds=time.perf_counter()-started,
                 hashes_before=before,hashes_after=hashes(models),selection='FINAL_EPOCH_WITH_MATCHED_OPTIMIZER')
    frozen_check(initial,summary['hashes_after'])
    atomic_json(out/'summary.json',summary)
    return summary


def coverage(frame,requested):
    valid=valid_frame(frame); counts=valid.groupby('parent_candidate_id').size()
    return dict(requested_parents=requested,unique_original_parents=frame.parent_candidate_id.nunique(),
                valid_children=len(valid),valid_parents=len(counts),complete_two_child_parents=int((counts==2).sum()),
                effective_sample_size_parent_clusters=len(counts),
                effective_sample_size_child_equal_weights=len(valid),
                duplicate_parent_fraction=0.,economic_failure_fraction=1-len(valid)/(2*requested),
                extinction_fraction=float(frame.failure_reason.fillna('').str.contains('ZERO_SURVIVING').sum())/(2*requested),
                nonpositive_consumption_fraction=float(frame.failure_reason.fillna('').str.contains('NONPOSITIVE').sum())/(2*requested),
                failure_reasons=frame.failure_reason.fillna('').value_counts().to_dict())


def history_row(models,args,k,train,val,seconds,device):
    v=valid_frame(val); r=v.normalized_G_at_prediction.to_numpy()
    row=dict(simulation_iteration=k,fc1_checkpoint_hash=module_hash(models['sdf_fc1'].fc1_model),
             valid_train_children=len(valid_frame(train)),valid_validation_children=len(v),simulation_seconds=seconds,
             GPU_peak_memory_GiB=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0.)
    for name,q in [('median',.5),('p90',.9),('p99',.99)]: row['G_'+name]=float(np.quantile(r,q)) if len(r) else float('nan')
    row.update(G_mean=float(r.mean()) if len(r) else float('nan'),G_max=float(r.max()) if len(r) else float('nan'))
    for forecast,target,prefix in [('Hatcf_t1','Hatc_cal_t1','Hatc'),('LnKF_t1','LnK_cal_t1','LnK')]:
        e=np.abs(v[forecast]-v[target])
        row[prefix+'_forecast_error_mean']=float(e.mean()); row[prefix+'_forecast_error_p90']=float(e.quantile(.9))
    row.update(FC1_C_R2=evaluate(models['sdf_fc1'],val,device).get('hatc_r2'),
               FC1_K_R2=evaluate(models['sdf_fc1'],val,device).get('lnk_r2'),
               Firm_mass_mean=float(v.surviving_mass.mean()),K_mean=float(v.K_cal_t1.mean()),C_mean=float(v.C_raw_t1.mean()))
    row.update({'train_'+k:v for k,v in coverage(train,args.train_parents).items() if not isinstance(v,dict)})
    row.update({'validation_'+k:v for k,v in coverage(val,args.validation_parents).items() if not isinstance(v,dict)})
    return row


def paired_change(previous,current):
    keys=['parent_candidate_id','child_branch']
    a=valid_frame(previous)[keys+['normalized_G_at_prediction']]
    b=valid_frame(current)[keys+['normalized_G_at_prediction']]
    match=a.merge(b,on=keys,suffixes=('_before','_after'))
    if len(match):
        match['delta_rho']=match.normalized_G_at_prediction_after-match.normalized_G_at_prediction_before
        match['relative_improvement']=1-match.normalized_G_at_prediction_after/np.maximum(match.normalized_G_at_prediction_before,1e-12)
    return match


def plot_results(root):
    history=pd.read_csv(root/'macro_fixed_point_history.csv')
    def line(filename,columns,ylabel):
        fig,ax=plt.subplots(figsize=(8,4.5))
        for col in columns:
            if col in history: ax.plot(history.simulation_iteration,history[col],marker='o',label=col)
        ax.set(xlabel='Simulation iteration',ylabel=ylabel); ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(root/'figures'/filename,dpi=160); plt.close(fig)
    (root/'figures').mkdir(exist_ok=True)
    line('g_residual_by_iteration.png',['G_mean','G_median','G_p90','G_p99'],'Normalized resource residual')
    line('fc1_function_movement.png',['FC1_C_function_move','FC1_K_function_move'],'Fixed probe RMS movement')
    line('fc1_two_cycle_diagnostics.png',['probe_two_cycle_distance','probe_direction_cosine'],'Two-cycle diagnostics')
    line('firm_mass_by_iteration.png',['Firm_mass_mean'],'Surviving firm count')
    line('capital_by_iteration.png',['K_mean'],'Capital')
    line('consumption_by_iteration.png',['C_mean'],'Raw consumption')
    line('valid_child_fraction_by_iteration.png',['train_economic_failure_fraction','validation_economic_failure_fraction'],'Failure fraction (original schedule)')
    line('economic_failure_by_iteration.png',['train_extinction_fraction','train_nonpositive_consumption_fraction'],'Failure fraction')
    line('default_investment_by_iteration.png',['default_rate','investment_rate','entry_rate'],'Decision rate')
    training=root/'fc1_training_history.csv'
    if training.exists():
        f=pd.read_csv(training)
        for filename,cols in [('fc1_train_validation_loss.png',['train_loss','validation_loss']),
                             ('fc1_train_validation_r2.png',['train_hatc_r2','validation_hatc_r2','train_lnk_r2','validation_lnk_r2'])]:
            fig,ax=plt.subplots(figsize=(9,4))
            for col in cols: ax.plot(np.arange(len(f))+1,f[col],label=col)
            ax.set(xlabel='Executed training epoch',ylabel='R2' if 'r2' in filename else 'Production loss')
            ax.legend(); fig.tight_layout(); fig.savefig(root/'figures'/filename,dpi=150); plt.close(fig)
    for col,label in [('Hatc','hatc'),('LnK','lnk')]:
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,k in zip(axes,(0,4,8)):
            path=root/f'simulation_{k:03d}'/'validation'/'observations.csv'
            if not path.exists(): continue
            f=valid_frame(pd.read_csv(path)); x=f['Hatcf_t1' if col=='Hatc' else 'LnKF_t1']; y=f[col+'_cal_t1']
            ax.scatter(x,y,s=6,alpha=.35); low=min(x.min(),y.min()); high=max(x.max(),y.max())
            ax.plot([low,high],[low,high],'k--'); ax.set(title=f'Simulation {k}',xlabel='FC1 prediction',ylabel='Calculated')
        fig.tight_layout(); fig.savefig(root/'figures'/f'fc1_{label}_prediction_vs_calculated.png',dpi=170); plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--train-parents',type=int,default=4096)
    p.add_argument('--validation-parents',type=int,default=1024); p.add_argument('--firms',type=int,default=200)
    p.add_argument('--iterations',type=int,default=8); p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--parent-batch-size',type=int,default=128); p.add_argument('--fit-parent-batch-size',type=int,default=128)
    p.add_argument('--seed',type=int,default=20261020); p.add_argument('--dynamic-parents',type=int,default=128)
    args=p.parse_args(); root=args.output; root.mkdir(parents=True,exist_ok=True)
    counters=dict(ROOT_SEARCH_CALLS=0,FAST_CLEARING_CALLS=0,FC2_FORWARD_CALLS=0,firm_forward_calls=0)
    install_guards(counters)
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    models,episode,optimizer,fc2=setup(args); episode._fc1_firms=args.firms
    models['policy_value'].register_forward_hook(lambda *unused: counters.update(firm_forward_calls=counters['firm_forward_calls']+1))
    initial=hashes(models)
    git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    atomic_json(root/'metadata.json',dict(args=vars(args),source_commit=git,initial_hashes=initial,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        representation='FINITE_FIRM_DIRECT_AGGREGATION',economic_equations_changed=False,
        initialization='Production stationary finite-firm draws; restricted Beta; K in [45,85]; fixed initial h0; t0 ledger calculated before FC1 child input.',
        z_coverage='Distinct stationary-law finite samples; no claim of varied theoretical stationary density shapes.',
        ess_definition='Independent valid parent clusters plus equal-weight valid-child count; siblings are not independent parents.',
        boundary_aware_continuum_fixed_point_test=False,optimizer='Fresh FC1 AdamW; no Firm/SDF optimizer used; no scheduler in this protocol.'))
    banks={}; seeds={}
    for split,n in [('train',args.train_parents),('validation',args.validation_parents)]:
        path=root/'banks'/f'{split}.pt'
        if path.exists():
            payload=torch.load(path,map_location='cpu',weights_only=False); bank=payload['state']; seed=payload['seed']
        else:
            bank,stats,seed=make_bank(models,episode,args,n,split)
            atomic_torch(path,dict(state=bank,seed=seed,fingerprint=digest_state(bank)))
            stats.to_csv(path.with_suffix('.csv'),index=False)
        banks[split]=bank; seeds[split]=seed
    train_ids=set(banks['train']['transition_path_id'].tolist()); val_ids=set(banks['validation']['transition_path_id'].tolist())
    if train_ids & val_ids: raise RuntimeError('train validation parent leakage')
    # Fixed innovations are keyed by stable economic/firm identities. Save the
    # common shock bank explicitly as well as its seed manifest.
    shock_bank={}
    for split,bank in banks.items():
        shock_bank[split]=torch.stack([_keyed_normal(seed=seeds[split]+11,path_id=bank['transition_path_id'],
            economic_time=torch.ones_like(bank['transition_path_id']),branch_id=b,
            identity=torch.zeros_like(bank['transition_path_id']),shock_type=1,dtype=torch.float32) for b in range(2)],dim=1)
    atomic_torch(root/'banks'/'common_child_shocks.pt',shock_bank)
    atomic_json(root/'banks'/'crn_manifest.json',dict(seeds=seeds,shock_bank_hash=digest_state(shock_bank),
        train_bank_hash=digest_state(banks['train']),validation_bank_hash=digest_state(banks['validation']),
        firm_and_entry_shocks='Production stable_firm_identity / stable_economic_identity; seeds fixed, never resampled.'))
    probe_path=root/'banks'/'macro_probe.pt'
    if probe_path.exists(): probe=torch.load(probe_path,map_location=episode.device,weights_only=False)
    else:
        bank=banks['validation']; count=min(1024,len(bank['x']))
        probe=torch.stack((bank['x'][:count],bank['x'][:count],bank['hatcf'][:count],bank['lnkf'][:count]),dim=1).to(episode.device)
        atomic_torch(probe_path,probe.cpu())
    history=[]; train_history=[]; stage_metrics=[]; probe_values=[]; previous_val=None
    # Rebuild the chronological record on resume. Existing data shards and
    # committed iteration checkpoints are immutable, not retrained.
    for k in range(args.iterations+1):
        if k>0:
            cp=torch.load(root/f'fc1_iteration_{k:03d}_complete.pt',map_location=episode.device,weights_only=False)
            models['sdf_fc1'].load_state_dict(cp['models']['sdf_fc1']); optimizer.load_state_dict(cp['optimizer']); restore_rng(cp['rng'])
        for model in models.values(): model.eval(); model.requires_grad_(False)
        frozen_check(initial,hashes(models))
        round_root=root/f'simulation_{k:03d}'; frames={}; total_seconds=0.
        for split in ('train','validation'):
            frames[split],seconds=simulate_bank(models,episode,args,banks[split],seeds[split],split,round_root/split)
            total_seconds+=seconds
            atomic_json(round_root/f'{split}_coverage.json',coverage(frames[split],len(banks[split]['x'])))
        train,val=frames['train'],frames['validation']
        if len(valid_frame(train))==0 or len(valid_frame(val))==0:
            atomic_json(root/'summary.json',dict(status='NO_LEGAL_FC1_LABELS',iteration=k,counters=counters))
            raise RuntimeError('NO_LEGAL_FC1_LABELS')
        row=history_row(models,args,k,train,val,total_seconds,episode.device)
        current_probe=predictions(models['sdf_fc1'],probe).cpu().numpy(); probe_values.append(current_probe)
        atomic_torch(round_root/'probe_predictions.pt',torch.from_numpy(current_probe))
        delta=current_probe-probe_values[-2] if k else np.zeros_like(current_probe)
        for j,name in enumerate(('FC1_C','FC1_K')):
            row[name+'_function_move']=float(np.sqrt(np.mean(delta[:,j]**2)))
            for q in (.5,.9,.99): row[name+f'_move_p{int(q*100)}']=float(np.quantile(np.abs(delta[:,j]),q))
        if k>=2:
            olddelta=probe_values[-2]-probe_values[-3]
            row['probe_direction_cosine']=float(np.sum(delta*olddelta)/max(np.linalg.norm(delta)*np.linalg.norm(olddelta),1e-30))
            row['probe_two_cycle_distance']=float(np.sqrt(np.mean((current_probe-probe_values[-3])**2)))
        if previous_val is not None:
            match=paired_change(previous_val,val); match.to_csv(round_root/'matched_valid_residual_changes.csv',index=False)
            row['matched_valid_children']=len(match)
            if len(match): row['matched_mean_delta_rho']=float(match.delta_rho.mean())
        previous_val=val
        v=valid_frame(val)
        row.update(default_rate=float((v.default_mass/v.firm_mass).mean()),
                   investment_rate=float((v.investment_mass/v.firm_mass).mean()),entry_rate=float((v.entry_mass/v.firm_mass).mean()))
        history.append(row); pd.DataFrame(history).to_csv(root/'macro_fixed_point_history.csv',index=False)
        stage_metrics.append(dict(iteration=k,stage='post_resimulation',**evaluate(models['sdf_fc1'],val,episode.device)))
        # Reuse exactly the same validation subset for all dynamic checkpoints.
        if k in (0,4,8):
            count=min(args.dynamic_parents,args.validation_parents)
            dynamic_bank=select_state(banks['validation'],slice(0,count))
            dynamic_frame,_=simulate_bank(models,episode,args,dynamic_bank,seeds['validation'],'validation',root/'dynamic'/f'phi_{k:03d}',horizon=5)
            by_time=[]
            for t,group in dynamic_frame.groupby('t'):
                legal=valid_frame(group)
                by_time.append(dict(t=int(t),valid_fraction=len(legal)/len(group),
                    G_mean=float(legal.normalized_G_at_prediction.mean()) if len(legal) else float('nan'),
                    mass_mean=float(legal.surviving_mass.mean()) if len(legal) else float('nan'),
                    K_mean=float(legal.K_cal_t1.mean()) if len(legal) else float('nan'),
                    C_mean=float(legal.C_raw_t1.mean()) if len(legal) else float('nan')))
            pd.DataFrame(by_time).to_csv(root/'dynamic'/f'phi_{k:03d}'/'by_time.csv',index=False)
        if k==args.iterations: break
        iteration=k+1; checkpoint=root/f'fc1_iteration_{iteration:03d}_complete.pt'
        if not checkpoint.exists():
            before_metrics=evaluate(models['sdf_fc1'],val,episode.device)
            summary=train_iteration(models,episode,optimizer,args,train,val,iteration,root/'training'/f'iteration_{iteration:03d}',initial)
            after_metrics=evaluate(models['sdf_fc1'],val,episode.device)
            for stage,values in [('pre_fit',before_metrics),('post_fit',after_metrics)]:
                stage_metrics.append(dict(iteration=k,stage=stage,**values))
            summary.update(pre_fit=before_metrics,post_fit=after_metrics)
            save_checkpoint(checkpoint,models,optimizer,iteration,args.epochs,
                dict(training_summary=summary,input_data_fingerprint=hashlib.sha256((round_root/'train'/'observations.csv').read_bytes()).hexdigest()))
        else:
            cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
            summary=cp['training_summary']
            for stage,key in [('pre_fit','pre_fit'),('post_fit','post_fit')]:
                stage_metrics.append(dict(iteration=k,stage=stage,**summary.get(key,{})))
        hist=pd.read_csv(root/'training'/f'iteration_{iteration:03d}'/'history.csv')
        train_history.extend(hist.to_dict('records')); pd.DataFrame(train_history).to_csv(root/'fc1_training_history.csv',index=False)
        pd.DataFrame(stage_metrics).to_csv(root/'fc1_fit_vs_resimulation.csv',index=False)
        plot_results(root)
        atomic_json(root/'progress.json',dict(training_iterations_completed=iteration,simulation_rounds_completed=k+1,
            hashes=hashes(models),counters=counters))
    frozen_check(initial,hashes(models)); counters['FC2_FORWARD_CALLS']=fc2['calls']
    common=valid_frame(pd.read_csv(root/'simulation_000'/'validation'/'observations.csv'))[['parent_candidate_id','child_branch']]
    for k in range(1,args.iterations+1):
        f=valid_frame(pd.read_csv(root/f'simulation_{k:03d}'/'validation'/'observations.csv'))
        common=common.merge(f[['parent_candidate_id','child_branch']],on=['parent_candidate_id','child_branch'])
    common_metrics=[]
    for k in range(args.iterations+1):
        f=pd.read_csv(root/f'simulation_{k:03d}'/'validation'/'observations.csv').merge(common,on=['parent_candidate_id','child_branch'])
        common_metrics.append(dict(iteration=k,n=len(f),G_mean=float(f.normalized_G_at_prediction.mean()),G_p90=float(f.normalized_G_at_prediction.quantile(.9))))
    pd.DataFrame(common_metrics).to_csv(root/'fixed_common_valid_history.csv',index=False)
    # Distribution coverage and difficulty refer to the fixed original bank.
    (root/'figures').mkdir(exist_ok=True)
    for split in ('train','validation'):
        bank_stats=pd.read_csv(root/'banks'/f'{split}.csv')
        fig,axes=plt.subplots(2,3,figsize=(12,7))
        for ax,name in zip(axes.flat,['mean_b','std_b','z_mean','z_std','K_total','K_hhi']):
            ax.hist(bank_stats[name],bins=35,color='#356ea2',alpha=.8); ax.set_xlabel(name)
        fig.suptitle(f'Fixed original {split} parent distribution coverage')
        fig.tight_layout(); fig.savefig(root/'figures'/f'{split}_parent_distribution_coverage.png',dpi=150); plt.close(fig)
        for k in range(args.iterations+1):
            obs=pd.read_csv(root/f'simulation_{k:03d}'/split/'observations.csv')
            merged=obs.merge(bank_stats,left_on='parent_candidate_id',right_on='parent_id',suffixes=('','_bank'))
            merged['b_mean_bin']=pd.cut(merged.mean_b,bins=[0,.15,.2,.25,.3,.35,.4,1.],include_lowest=True)
            difficulty=merged.groupby('b_mean_bin',observed=True).agg(
                candidate_children=('parent_candidate_id','size'),valid_children=('economic_state_valid','sum'),
                G_mean=('normalized_G_at_prediction','mean'))
            difficulty.to_csv(root/f'simulation_{k:03d}'/f'{split}_beta_difficulty.csv')
            bank_stats.to_csv(root/f'simulation_{k:03d}'/f'{split}_parent_coverage.csv',index=False)
    for col in ('G_mean','valid_fraction','mass_mean','K_mean','C_mean'):
        fig,ax=plt.subplots(figsize=(7,4))
        for k in (0,4,8):
            p=root/'dynamic'/f'phi_{k:03d}'/'by_time.csv'
            if p.exists():
                f=pd.read_csv(p); ax.plot(f.t,f[col],marker='o',label=f'phi_{k}')
        ax.set(xlabel='Economic time',ylabel=col); ax.legend(); fig.tight_layout()
        fig.savefig(root/'figures'/f'dynamic_{col}.png',dpi=150); plt.close(fig)
    first,last=history[0],history[-1]
    improved=last['G_mean']<first['G_mean']
    coverage_stable=last['validation_economic_failure_fraction']<=first['validation_economic_failure_fraction']+.05
    classification='INCONCLUSIVE'
    if len(common)>0:
        ratio=common_metrics[-1]['G_mean']/max(common_metrics[0]['G_mean'],1e-12)
        classification='CONVERGING' if ratio<.8 and coverage_stable else ('DIVERGING' if ratio>1.2 else 'STAGNATING')
        if history[-1].get('probe_direction_cosine',1)<-.8 and history[-1].get('probe_two_cycle_distance',float('inf'))<max(history[-1]['FC1_C_function_move'],1e-12)*.25:
            classification='OSCILLATING'
    all_label=all(r['valid_train_children']==args.train_parents*2 and r['valid_validation_children']==args.validation_parents*2 for r in history)
    summary=dict(status='COMPLETED',FIRM_FROZEN=True,SDF_FROZEN=True,ROOT_SEARCH_USED=False,FAST_CLEARING_USED=False,
        MACRO_ITERATIONS_COMPLETED=args.iterations,SIMULATION_ROUNDS_COMPLETED=len(history),
        TRAIN_PARENT_TARGET=args.train_parents,VALIDATION_PARENT_TARGET=args.validation_parents,
        FC1_ACTUALLY_TRAINED_EACH_ROUND=all(pd.read_csv(root/'training'/f'iteration_{i:03d}'/'history.csv').epoch.max()==args.epochs for i in range(1,args.iterations+1)),
        MACRO_LABELS_REGENERATED_EACH_ROUND=True,DATA_GENERATION_TARGET_REACHED=all_label,
        FC1_TRAINING_COMPLETED=True,MARKET_CLEARING_RESIDUAL_IMPROVED=improved,
        ECONOMIC_STATE_COVERAGE_STABLE=coverage_stable,MACRO_FIXED_POINT_EVIDENCE=classification,
        PARENT_DISTRIBUTION_COVERAGE_REPORTED=True,EFFECTIVE_SAMPLE_SIZE_REPORTED=True,
        counters=counters,hashes_before=initial,hashes_after=hashes(models),
        initial_residual=first,final_residual=last,common_valid_child_count=len(common),source_commit=git)
    if counters['ROOT_SEARCH_CALLS'] or fc2['calls']: raise RuntimeError('forbidden operator called')
    atomic_json(root/'summary.json',summary); plot_results(root)
    report=root/'reports'/'fc1_self_consistent_macro_fixed_point_20261020.md'
    report.parent.mkdir(exist_ok=True)
    report.write_text('# FC1 self-consistent macro fixed-point experiment\n\n'
        'Finite-firm production aggregation; restricted Beta support. No root search.\n\n'
        'Frozen Firm/SDF, independently grouped train/validation and fixed keyed CRN.\n\n'
        'Labels are conditioned on valid economic states; single valid children are allowed for FC1 only.\n\n'
        'Initial z varies through distinct stationary finite samples, not through changed innovation parameters.\n\n'
        'The same physical resource ledger produces C/K and G. Survivor mass is distinguished from pre-decision participating firm count.\n\n'
        'Fixed common-valid residual history and original-schedule coverage are separate artifacts.\n\n'
        'Function movement and finite-sample evidence do not prove a global fixed point or input sufficiency.\n\n'
        '```json\n'+json.dumps(summary,indent=2,default=str)+'\n```\n')
    print(json.dumps(summary,default=str),flush=True)


if __name__=='__main__':
    try: main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
