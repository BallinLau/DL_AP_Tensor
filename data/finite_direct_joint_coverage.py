"""Macro-aware extension of the existing finite-direct Parent Bank schema.

Only proposals/selection change. All certification calls the original h-star,
production ledger and Grid BP. No label can be generated from a proposed h.
"""
from collections import Counter,defaultdict
import csv
import json
import time
from dataclasses import dataclass,asdict,replace
from pathlib import Path
import hashlib
import math
import numpy as np
import torch
from config import Config
from analysis.nested_equilibrium_utils import atomic_torch_save,atomic_write_json,stable_json_hash
from analysis.run_fc1_self_consistent_macro import select_state
from data.finite_direct_sampling import state_hash,file_hash,capital_groups
from data.simulate_ts_parallel import _keyed_uniform,_keyed_normal
from data.finite_direct_initial_hstar import calibrate_initial,InitialCalibrationBlocked
from analysis.finite_direct_continuation_state import safe_load

VERSION='H50_BETA_JOINT_COVERAGE_V2_DIRECTIONAL_REFILL'


@dataclass(frozen=True)
class JointCoverageConfig:
    enabled:bool=True
    seed:int=2026102141
    families:tuple=('low','medium','high','broad','u_shaped','bimodal','upper_tail')
    beta_parameters:tuple=((2.,12.),(8.,8.),(12.,2.),(1.,1.),(.4,.4))
    mixture_weight:float=.5
    upper_tail_weight:float=.15
    k_scales:tuple=(.25,.5,1.,2.,4.)
    x_quantiles:tuple=(.01,.25,.5,.75,.99)
    rank_modes:tuple=('independent','K_positive','K_negative','z_positive','z_negative')
    h_bounds:tuple=(-2.2,.4)
    ell_bounds:tuple=(2.3,6.)
    n_h_bins:int=8
    n_ell_bins:int=8
    bin_cap:int=256
    candidate_batch_size:int=16
    max_candidates:int=32768
    per_lineage_reuse_limit:int=1
    recent_fraction:float=.7
    conditional_min_count:int=8
    def validate(self):
        if not self.enabled:raise ValueError('this version requires explicit joint coverage; legacy plan needs separate review')
        if min(self.bin_cap,self.candidate_batch_size,self.max_candidates,self.per_lineage_reuse_limit)<1:
            raise ValueError('coverage budgets must be bounded positive integers')
        if tuple(self.families)!=('low','medium','high','broad','u_shaped','bimodal','upper_tail'):
            raise ValueError('declared distribution families changed')
        if len(self.beta_parameters)!=5 or any(a<=0 or b<=0 for a,b in self.beta_parameters):raise ValueError('invalid Beta parameters')
        if not self.k_scales or any(not math.isfinite(k) or k<=0 for k in self.k_scales):raise ValueError('invalid K tier')
        if not self.x_quantiles or any(not 0<q<1 for q in self.x_quantiles):raise ValueError('invalid stationary x quantile')
        if not self.rank_modes or any(m not in ('independent','K_positive','K_negative','z_positive','z_negative') for m in self.rank_modes):raise ValueError('invalid rank mode')
        if not 0<=self.mixture_weight<=1 or not 0<=self.upper_tail_weight<=1:raise ValueError('invalid mixture')
        if not 0<=self.recent_fraction<=1:raise ValueError('invalid Recent share')
        if self.h_bounds[0]>=self.h_bounds[1] or self.ell_bounds[0]>=self.ell_bounds[1] or min(self.n_h_bins,self.n_ell_bins)<1:raise ValueError('invalid joint bins')
    def edges(self):
        return np.linspace(*self.h_bounds,self.n_h_bins+1),np.linspace(*self.ell_bounds,self.n_ell_bins+1)


def joint_bin(h,ell,cfg):
    if not math.isfinite(h) or not math.isfinite(ell):return None
    he,ke=cfg.edges()
    def axis(v,e):
        if v<e[0]:return -1
        if v>e[-1]:return len(e)-1
        return min(int(np.searchsorted(e,v,side='right')-1),len(e)-2)
    return axis(h,he),axis(ell,ke)


def bin_direction(cell,cfg):
    if cell is None:return 'INVALID'
    h,k=cell
    return ('H_LOW' if h<0 else 'H_HIGH' if h>=cfg.n_h_bins else 'H_IN')+'/'+('ELL_LOW' if k<0 else 'ELL_HIGH' if k>=cfg.n_ell_bins else 'ELL_IN')


def source_allocation(total,empirical_eligible,beta_eligible):
    """Capacity counts mean independent eligible units, never raw row count."""
    if min(total,empirical_eligible,beta_eligible)<0:raise ValueError('negative capacity')
    emp=min(total//2,empirical_eligible);beta=min(total-emp,beta_eligible)
    return dict(target=total,empirical=emp,beta=beta,underfilled=total-emp-beta,
        target_empirical=total//2,beta_fallback=max(0,beta-(total-total//2)))


def beta_values(family,n,rng,cfg):
    if family in cfg.families[:5]:
        a,b=cfg.beta_parameters[cfg.families[:5].index(family)];return rng.beta(a,b,n)
    weight=cfg.mixture_weight if family=='bimodal' else cfg.upper_tail_weight
    if family not in ('bimodal','upper_tail'):raise ValueError('unknown family')
    high=rng.random(n)<weight
    return np.where(high,rng.beta(*cfg.beta_parameters[2],n),rng.beta(*cfg.beta_parameters[0],n))


def propose_beta_batch(reference_bank,*,start,count,outer,cfg,design_points=None):
    """Vectorized exogenous draws with original calibrated stationary laws.

    Use the existing immutable initial Bank layout and relative capital weights.
    New ids/seeds provide new x/z/eta/i draws; there is no per-firm NN call.
    """
    cfg.validate();base=reference_bank['state'];nbase=len(base['x'])
    ids=torch.arange(start,start+count,dtype=torch.long)
    indices=(ids*97+cfg.seed)%nbase;s=select_state(base,indices,torch.device('cpu'))
    if not bool(s['alive'].all()):raise ValueError('coverage requires genuine initial-cohort Bank')
    B,N=s['K'].shape
    pid=10_000_000+outer*100_000+ids
    slots=torch.arange(N)[None,:].expand(B,N);paths=pid[:,None].expand(B,N)
    zeros=torch.zeros_like(paths)
    uniform=lambda typ:_keyed_uniform(seed=cfg.seed,path_id=paths,economic_time=zeros,
        branch_id=zeros,identity=slots,shock_type=typ,dtype=torch.float64)
    normal=_keyed_normal(seed=cfg.seed,path_id=paths,economic_time=zeros,branch_id=zeros,
        identity=slots,shock_type=401,dtype=torch.float64)
    s['z']=(float(Config.ZBAR)+float(Config.SIGMA_Z)/math.sqrt(1-float(Config.RHO_Z)**2)*normal).float()
    s['eta']=(uniform(402)<float(Config.ZETA)).float()
    s['i']=(uniform(403)*float(Config.I_THRESHOLD)).float()
    specs=[];bs=[]
    for j,index in enumerate(ids.tolist()):
        rng=np.random.default_rng(cfg.seed+outer*100003+index*193)
        family=cfg.families[index%len(cfg.families)] if design_points is None else cfg.families[min(len(cfg.families)-1,int(float(design_points[j,3])*len(cfg.families)))]
        bs.append(beta_values(family,N,rng,cfg))
        scale=cfg.k_scales[(index//len(cfg.families)+index)%len(cfg.k_scales)]
        q=cfg.x_quantiles[int(rng.integers(len(cfg.x_quantiles)))]
        zx=float(torch.distributions.Normal(0.,1.).icdf(torch.tensor(q,dtype=torch.float64)))
        s['x'][j]=float(Config.XBAR)+float(Config.SIGMA_X)/math.sqrt(1-float(Config.RHO_X)**2)*zx
        total=float(s['K'][j].sum())*scale
        s['K'][j]*=scale
        specs.append(dict(parent_id=int(pid[j]),split='train',sampling_source='BETA',source_type='BETA_JOINT_ECONOMIC_COVERAGE',
            source_outer=outer,source_lineage=int(pid[j]),source_snapshot_time=0,
            source_pool_fingerprint=stable_json_hash(dict(generator=VERSION,config=asdict(cfg))),coverage_family=family,
            sampling_weight=1.,reuse_count=1,base_source_parent=int(base['transition_path_id'][indices[j]]),
            K_scale=scale,K_family='BOUNDED_LOG_TIERS',K_total_requested=total,
            x_quantile=q,rank_mode=(cfg.rank_modes[int(rng.integers(len(cfg.rank_modes)))] if design_points is None else
                cfg.rank_modes[min(len(cfg.rank_modes)-1,int(float(design_points[j,4])*len(cfg.rank_modes)))]),
            child_shock_keys=[dict(path_id=int(pid[j]),economic_time=1,branch_id=b,
                transition_seed=cfg.seed+11,entry_seed=cfg.seed+29) for b in (0,1)]))
    b=torch.tensor(np.array(bs),dtype=torch.float32)
    # Rank-permute only b; z and K marginals are not changed by coupling.
    for mode,field,reverse in [('K_positive','K',False),('K_negative','K',True),('z_positive','z',False),('z_negative','z',True)]:
        rows=torch.tensor([j for j,p in enumerate(specs) if p['rank_mode']==mode],dtype=torch.long)
        if not len(rows):continue
        order=torch.argsort(s[field][rows],dim=1,stable=True);values=b[rows].sort(dim=1).values
        if reverse:values=values.flip(1)
        assigned=torch.empty_like(values);assigned.scatter_(1,order,values);b[rows]=assigned
    s['b']=b;s['bp']=b.clone();s['K_birth']=s['K'].clone();s['lnkf']=s['K'].sum(1).log()
    s['transition_path_id']=pid
    for j,p in enumerate(specs):
        values=b[j].numpy();z=s['z'][j].numpy();k=s['K'][j].numpy()
        p.update(b_mean=float(values.mean()),b_std=float(values.std()),
            **{f'b_p{int(q*100)}':float(np.quantile(values,q)) for q in (.1,.5,.9,.99)},
            high_b_share=float((values>.5).mean()),corr_b_z=float(np.corrcoef(values,z)[0,1]),
            corr_b_K=float(np.corrcoef(values,k)[0,1]),K_total_realized=float(k.sum()))
    return s,specs


class JointReservoir:
    """Bounded metadata references, first joint bin then family diversity.

    No tensors retained here. Scarce-bin newcomers replace the largest bin's
    most represented family; seeded priorities resolve ties reproducibly.
    """
    def __init__(self,capacity,cfg):
        self.capacity=capacity;self.cfg=cfg;self.buckets=defaultdict(list);self.count=0
        self.occupancy_by_bin=Counter();self.occupancy_by_bin_family=Counter()
        self.families_by_bin={}
        self.eligible_count_by_bin=Counter();self.cap_limited_rejections=0
    def add(self,record):
        bin_=tuple(record['coverage_bin']);family=record['coverage_family'];key=(bin_,family)
        r=dict(record,priority=int(hashlib.sha256(f'{self.cfg.seed}:{record["parent_id"]}'.encode()).hexdigest()[:16],16))
        bins=self.occupancy_by_bin;self.eligible_count_by_bin[bin_]+=1
        cap_hit=bins[bin_]>=self.cfg.bin_cap
        if cap_hit:self.cap_limited_rejections+=1  # rejected newcomer OR displaced eligible incumbent
        if bins[bin_]>=self.cfg.bin_cap or self.count>=self.capacity:
            victim_bin=bin_ if bins[bin_]>=self.cfg.bin_cap else max(bins,key=lambda b:(bins[b],b))
            if victim_bin!=bin_ and bins[bin_]>=bins[victim_bin]:return False
            # Empirical tests may supply other family names; register only the
            # bounded occupied family index, never rescan every bucket.
            candidates=[(victim_bin,f) for f in self.families_by_bin.get(victim_bin,()) if self.occupancy_by_bin_family[victim_bin,f]>0]
            victim=max(candidates,key=lambda k:(len(self.buckets[k]),k[1]))
            if victim_bin==bin_ and len(self.buckets[key])>=len(self.buckets[victim]):
                victim=key
                worst=min(self.buckets[victim],key=lambda x:x['priority'])
                if r['priority']<=worst['priority']:
                    return False
            worst=min(self.buckets[victim],key=lambda x:x['priority']);self.buckets[victim].remove(worst);self.count-=1
            self.occupancy_by_bin[victim_bin]-=1;self.occupancy_by_bin_family[victim]-=1
        self.buckets[key].append(r);self.count+=1
        self.occupancy_by_bin[bin_]+=1;self.occupancy_by_bin_family[key]+=1
        self.families_by_bin.setdefault(bin_,set()).add(family)
        return True
    def selected(self,n):
        pools={k:sorted(v,key=lambda x:x['priority'],reverse=True) for k,v in self.buckets.items() if v}
        result=[]
        while pools and len(result)<n:
            for k in sorted(pools,key=lambda k:(len(pools[k]),k)):
                if len(result)==n:break
                result.append(pools[k].pop(0))
            pools={k:v for k,v in pools.items() if v}
        return result


def eligible_templates(templates,*,outer,cfg,committed):
    """Unknown historical node accounting is NOT promoted to certified True."""
    selected=[];reasons=Counter();used=Counter();seen=set()
    if not committed:return [],{'UNCOMMITTED_POOL':sum(1 for _ in templates)}
    for t in templates:
        if t.get('split')!='train':reasons['NOT_TRAIN']+=1;continue
        if t.get('source_outer',outer)>=outer:reasons['FUTURE_SOURCE']+=1;continue
        if t.get('post_decision_accounting_valid') is not True:reasons['ACCOUNTING_UNVERIFIED_OR_INVALID']+=1;continue
        lineage=t.get('cluster_id');tid=t.get('template_id')
        if lineage is None or tid in seen:reasons['INVALID_OR_DUPLICATE_ID']+=1;continue
        b=np.asarray(t.get('b_sorted',[]),float)
        if not len(b) or not np.isfinite(b).all() or (b<0).any() or (b>1).any() or len(t.get('b_by_capital_quartile',[]))!=4:
            reasons['INCOMPATIBLE_TEMPLATE']+=1;continue
        if used[lineage]>=cfg.per_lineage_reuse_limit:reasons['LINEAGE_REUSE_CAP']+=1;continue
        if len(selected)>=cfg.max_candidates:reasons['BOUNDED_ELIGIBILITY_CAP']+=1;continue
        used[lineage]+=1;seen.add(tid);selected.append(t)
    return selected,dict(reasons)


def certify_batch(sim,state,*,outer,cfg,economic,crn):
    """Batch-first; split only failing batches to preserve diagnostic evidence."""
    try:
        processed,records,valid,audit=calibrate_initial(sim,state,outer=outer,split='train',economic=economic,crn=crn)
        return [(j,processed,records[j],audit[j]) for j in range(len(records))],[]
    except InitialCalibrationBlocked as exc:
        if len(state['x'])==1:
            return [],[dict(parent_id=int(state['transition_path_id'][0]),reason='STRICT_HSTAR_FAILED',details=exc.audit)]
        mid=len(state['x'])//2;success=[];fail=[]
        for lo,hi in ((0,mid),(mid,len(state['x']))):
            part=select_state(state,slice(lo,hi),sim.device)
            a,b=certify_batch(sim,part,outer=outer,cfg=cfg,economic=economic,crn=crn)
            success.extend((j+lo,p,r,audit) for j,p,r,audit in a);fail.extend(b)
        return success,fail


def _empirical_proposal(reference,templates,*,start,outer,cfg,pool_fingerprint):
    indices=torch.arange(start,start+len(templates))%len(reference['state']['x'])
    s=select_state(reference['state'],indices,torch.device('cpu'));specs=[]
    for j,t in enumerate(templates):
        rng=np.random.default_rng(cfg.seed+outer*100003+start+j);k=s['K'][j].numpy();groups=capital_groups(k)
        b=np.empty_like(k);u=rng.random(len(k))
        for q in range(4):
            mask=groups==q;v=t['b_by_capital_quartile'][q]
            if len(v)<cfg.conditional_min_count:v=t['b_sorted']
            b[mask]=np.asarray(v)[np.floor(u[mask]*len(v)).astype(int)]
        s['b'][j]=torch.from_numpy(b);s['bp'][j]=s['b'][j]
        pid=20_000_000+outer*100_000+start+j;s['transition_path_id'][j]=pid
        specs.append(dict(parent_id=pid,split='train',sampling_source='EMPIRICAL',source_type='DYNAMIC_EMPIRICAL',
            source_outer=t['source_outer'],source_lineage=t['cluster_id'],source_snapshot_time=t['source_time'],
            source_pool_fingerprint=pool_fingerprint,coverage_family='empirical',sampling_weight=1.,reuse_count=1,
            template_id=t['template_id'],K_scale=1.,K_family='SOURCE_INITIAL_K',
            K_total_requested=float(k.sum()),K_total_realized=float(k.sum()),
            b_mean=float(b.mean()),b_std=float(b.std()),
            **{f'b_p{int(q*100)}':float(np.quantile(b,q)) for q in (.1,.5,.9,.99)},
            high_b_share=float((b>.5).mean()),corr_b_z=float(np.corrcoef(b,s['z'][j].numpy())[0,1]),
            corr_b_K=float(np.corrcoef(b,k)[0,1]),
            child_shock_keys=[dict(path_id=pid,economic_time=1,branch_id=branch,
                transition_seed=cfg.seed+11,entry_seed=cfg.seed+29) for branch in (0,1)]))
    return s,specs


@torch.inference_mode()
def build_joint_bank(reference,templates,*,outer,target,cfg,output,simulator_factory,economic,
                     committed=True,pool_fingerprint='CERTIFIED_SOURCE_POOL'):
    """Atomic, resumable candidate shards + bounded reference reservoir.

    The selected object IS the original Parent Bank schema. No parallel
    economic simulator or unbounded candidate tensor pool is introduced.
    """
    cfg.validate();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    from analysis.run_fc1_self_consistent_macro import hashes
    identity=dict(version=VERSION,config=asdict(cfg),outer=outer,target=target,
        reference_bank=reference['fingerprint'],pool=pool_fingerprint,economic=economic,
        models=hashes(simulator_factory(1).models),committed=committed,
        source_sha256=file_hash(__file__))
    identity['production_sha256']={name:file_hash(Path(__file__).resolve().parents[1]/name) for name in (
        'data/finite_direct_initial_hstar.py','data/market_clearing.py','data/finite_direct_runtime.py',
        'data/simulate_ts_parallel.py','data/simulate_ts.py','data/entry.py','data/simulation_forward.py',
        'models/policy_value.py','models/firm_derived.py','analysis/run_fc1_self_consistent_macro.py')}
    identity=json.loads(json.dumps(identity));identity_path=output/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:raise ValueError('coverage cache identity mismatch')
    atomic_write_json(identity_path,identity)
    final=output/'selected_manifest.json'
    if final.exists():
        m=json.loads(final.read_text())
        if m['identity']!=identity:raise ValueError('selected coverage manifest identity mismatch')
        for k in ('bank','schedule'):
            if file_hash(output/m[k]['file'])!=m[k]['sha256']:raise ValueError('immutable coverage Bank corruption')
        bank=safe_load(output/m['bank']['file']);schedule=safe_load(output/m['schedule']['file'])
        if state_hash(bank['state'])!=bank['fingerprint']:raise ValueError('coverage state corruption')
        if bank['schedule_fingerprint']!=schedule['fingerprint'] or schedule['fingerprint']!=stable_json_hash(schedule['parents']):raise ValueError('coverage schedule corruption')
        return bank,schedule,m['stats']
    eligible,rejected=eligible_templates(templates,outer=outer,cfg=cfg,committed=committed)
    # Source rank/order is deterministic. Reuse is bounded before any drawing.
    recent=[t for t in eligible if t.get('source_outer')==outer-1];archive=[t for t in eligible if t.get('source_outer')!=outer-1]
    goal=min(target//2,cfg.max_candidates);nr=min(len(recent),int(goal*cfg.recent_fraction));na=min(len(archive),goal-nr)
    nr=min(len(recent),goal-na)
    eligible=recent[:nr]+archive[:na]
    empirical=[];beta=JointReservoir(target,cfg)
    refill=JointReservoir(2*target,replace(cfg,bin_cap=2*target));counters=Counter();timers=Counter()
    started=time.monotonic();audit_path=output/'candidate_acceptance.csv'
    columns=['parent_id','source_type','coverage_family','source_lineage','source_outer','source_snapshot_time',
        'K_scale','K_total_requested','K_total_realized','h_belief_proposed','ell_proposed','h_cal','ell_cal',
        'root_status','root_residual','hstar_pass','ledger_valid','economic_valid','coverage_bin','overflow_direction',
        'b_mean','b_std','b_p10','b_p50','b_p90','b_p99','high_b_share','corr_b_z','corr_b_K','rejection_reason']
    with audit_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        offset=0
        for source in ('DYNAMIC_EMPIRICAL','BETA_COVERAGE'):
            limit=len(eligible) if source=='DYNAMIC_EMPIRICAL' else cfg.max_candidates-len(eligible)
            for start in range(0,limit,cfg.candidate_batch_size):
                count=min(cfg.candidate_batch_size,limit-start);file=output/f'candidates_{offset:06d}.pt';side=file.with_suffix('.sha.json')
                if file.exists() or side.exists():
                    if not file.exists() or not side.exists() or file_hash(file)!=json.loads(side.read_text())['sha256']:
                        raise ValueError('uncommitted/corrupt coverage candidate shard')
                    saved=safe_load(file)
                    if saved['identity']!=identity:raise ValueError('stale candidate shard')
                    state,specs,certified,failures=saved['state'],saved['specs'],saved['certified'],saved['failures']
                else:
                    clock=time.monotonic()
                    state,specs=(_empirical_proposal(reference,eligible[start:start+count],start=start,outer=outer,cfg=cfg,
                        pool_fingerprint=pool_fingerprint) if source=='DYNAMIC_EMPIRICAL' else
                        propose_beta_batch(reference,start=start,count=count,outer=outer,cfg=cfg))
                    timers['candidate_generation_seconds']+=time.monotonic()-clock;clock=time.monotonic()
                    result,failures=certify_batch(simulator_factory(count),state,outer=outer,cfg=cfg,economic=economic,crn={'seed':cfg.seed})
                    certified={int(specs[j]['parent_id']):dict(record=r,audit=a) for j,_,r,a in result}
                    timers['hstar_seconds']+=time.monotonic()-clock
                    atomic_torch_save(file,dict(identity=identity,state=state,specs=specs,certified=certified,failures=failures))
                    atomic_write_json(side,dict(sha256=file_hash(file)))
                shard_sha=json.loads(side.read_text())['sha256']
                for j,p in enumerate(specs):
                    pid=p['parent_id'];value=certified.get(pid);r=value['record'] if value else {};a=value['audit'] if value else {}
                    h=r.get('Hatc_cal_t1',float('nan'));ell=r.get('LnK_cal_t1',float('nan'))
                    ledger=math.isfinite(h) and math.isfinite(ell) and r.get('C_raw_t1',0)>0 and r.get('K_cal_t1',0)>0
                    valid=bool(value) and ledger and bool(r.get('economic_state_valid',False))
                    bins=joint_bin(h,ell,cfg)
                    row=dict(p,h_belief_proposed=float(state['hatcf'][j]),ell_proposed=float(state['lnkf'][j]),
                        h_cal=h,ell_cal=ell,hstar_pass=bool(value),ledger_valid=ledger,economic_valid=valid,
                        root_status='ROOT_FOUND' if value else 'STRICT_HSTAR_FAILED',root_residual=a.get('normalized_residual'),
                        coverage_bin=bins,overflow_direction=bin_direction(bins,cfg),rejection_reason='' if valid else ('ROOT_FAILED' if not value else 'CURRENT_NODE_ECONOMIC_INVALID'))
                    writer.writerow({k:json.dumps(row.get(k)) if isinstance(row.get(k),(dict,list,tuple)) else row.get(k) for k in columns})
                    counters['proposed']+=1;counters['hstar_pass']+=bool(value);counters['ledger_valid']+=ledger;counters['economic_valid']+=valid
                    if valid:
                        ref=dict(p,h_cal=h,ell_cal=ell,coverage_bin=bins,artifact=str(file),artifact_sha256=shard_sha,row=j)
                        if source=='DYNAMIC_EMPIRICAL':empirical.append(ref)
                        else:beta.add(ref);refill.add(ref)
                offset+=count
                elapsed=time.monotonic()-started
                progress=dict(event='JOINT_CANDIDATE_PROGRESS',outer=outer,source=source,
                    proposed=counters['proposed'],economic_valid=counters['economic_valid'],
                    max_candidates=cfg.max_candidates,elapsed_seconds=elapsed,
                    candidates_per_second=counters['proposed']/max(elapsed,1e-9),
                    estimated_remaining_seconds=(cfg.max_candidates-counters['proposed'])*elapsed/max(counters['proposed'],1),
                    adaptive_early_stop=False)
                atomic_write_json(output/'progress.json',progress)
                print(json.dumps(progress),flush=True);stream.flush()
    allocation=source_allocation(target,len(empirical),refill.count)
    stage1=beta.selected(allocation['beta']);chosen={r['parent_id'] for r in stage1}
    extras=[r for r in refill.selected(2*target) if r['parent_id'] not in chosen]
    occupancy=Counter(tuple(r['coverage_bin']) for r in stage1);refilled=[]
    # Stage2 only references already certified distinct candidates; never draw
    # a repeated Parent or increase the candidate/economic budget.
    groups=defaultdict(list)
    for r in extras:groups[tuple(r['coverage_bin'])].append(r)
    while groups and len(stage1)+len(refilled)<allocation['beta']:
        cell=min(groups,key=lambda b:(occupancy[b],b))
        r=groups[cell].pop(0);r=dict(r,selection_stage='CAPACITY_REFILL');refilled.append(r);occupancy[cell]+=1
        if not groups[cell]:del groups[cell]
    selected=empirical[:allocation['empirical']]+stage1+refilled
    if not selected:raise ValueError('NO_ROOT_CERTIFIED_ECONOMIC_COVERAGE_PARENTS; see candidate_acceptance.csv')
    # Materialize only the FINAL bounded Bank, never all candidate tensors.
    first=safe_load(selected[0]['artifact'])['state'];M=len(selected);B=len(first['x'])
    state={k:torch.empty((M,*v.shape[1:]),dtype=v.dtype) if torch.is_tensor(v) and v.ndim and len(v)==B else v for k,v in first.items()}
    grouped=defaultdict(list)
    for j,r in enumerate(selected):grouped[r['artifact']].append((j,r['row']))
    for path,positions in grouped.items():
        expected=next(r['artifact_sha256'] for r in selected if r['artifact']==path)
        if file_hash(path)!=expected:raise ValueError('candidate artifact changed before selection commit')
        original=safe_load(path)['state'];out=torch.tensor([i for i,_ in positions]);ids=torch.tensor([i for _,i in positions])
        for k,v in original.items():
            if torch.is_tensor(v) and v.ndim and len(v)==len(original['x']):state[k][out]=v[ids]
    parents=[{k:v for k,v in r.items() if k not in ('artifact','artifact_sha256','row','priority')} for r in selected]
    schedule=dict(version=VERSION,split='train',seed=cfg.seed,parents=parents,fingerprint=stable_json_hash(parents),
        empirical_pool_fingerprint=pool_fingerprint)
    bank=dict(state=state,seed=cfg.seed,fingerprint=state_hash(state),schedule_fingerprint=schedule['fingerprint'],
        bank_version=f'H50_JOINT_outer_{outer}',empirical_pool_fingerprint=pool_fingerprint,
        coverage_config=asdict(cfg),source_manifest=parents)
    stats=dict(counters,allocation=allocation,empirical_pre_root_eligible=len(eligible),empirical_rejections=rejected,
        actual_initial_source_ratio=dict(Counter(r['source_type'] for r in parents)),
        selected_families=dict(Counter(r['coverage_family'] for r in parents)),
        unique_macro_bins=len({tuple(r['coverage_bin']) for r in parents}),
        unique_initial_parents=len({r['parent_id'] for r in parents}),
        occupied_bin_ratio=len({tuple(r['coverage_bin']) for r in parents if bin_direction(tuple(r['coverage_bin']),cfg)=='H_IN/ELL_IN'})/(cfg.n_h_bins*cfg.n_ell_bins),
        timers=dict(timers),total_seconds=time.monotonic()-started,
        proposed_h_cal='NOT_DEFINED_UNTIL_STRICT_ROOT',h_bounds=cfg.h_bounds,ell_bounds=cfg.ell_bounds,
        h_edges=cfg.edges()[0].tolist(),ell_edges=cfg.edges()[1].tolist(),
        selection_is_population_sample=False,gpu_performance='NOT_TESTED')
    beta_legal=sum(beta.eligible_count_by_bin.values());beta_goal=target-allocation['empirical']
    capped_capacity=sum(min(n,cfg.bin_cap) for n in beta.eligible_count_by_bin.values())
    stats['capacity_audit']=dict(occupied_bins=len(beta.eligible_count_by_bin),
        eligible_count_by_bin={str(k):v for k,v in beta.eligible_count_by_bin.items()},
        retained_count_by_bin={str(k):v for k,v in occupancy.items()},
        cap_limited_rejections=beta.cap_limited_rejections,
        economic_invalid_rejections=counters['hstar_pass']-counters['economic_valid'],root_failed_rejections=counters['proposed']-counters['hstar_pass'],
        reachable_capacity_under_current_caps=capped_capacity,target_bank_size=target,actual_bank_size=len(selected),
        stage1_bank_size=len(empirical[:allocation['empirical']])+len(stage1),capacity_refill_count=len(refilled),
        capacity_shortfall=max(0,min(beta_goal,beta_legal)-min(beta_goal,capped_capacity)),
        economic_shortfall=max(0,target-len(empirical)-beta_legal),candidate_budget_exhausted=counters['proposed']>=cfg.max_candidates)
    # Bounded scalar-only diagnostics, never retained simulation tensors.
    import pandas as pd
    audit=pd.read_csv(audit_path);phase_rows=[];grid_rows=[]
    phases={'PROPOSED':audit,'HSTAR_VALID':audit[audit.hstar_pass],
        'ECONOMIC_VALID':audit[audit.economic_valid],
        'SELECTED':audit[audit.parent_id.isin([r['parent_id'] for r in parents])]}
    for phase,frame in phases.items():
        row=dict(phase=phase,count=len(frame))
        for field in ('h_belief_proposed','ell_proposed','h_cal','ell_cal','b_mean'):
            values=frame[field].to_numpy(dtype=float);values=values[np.isfinite(values)]
            if phase=='PROPOSED' and field in ('h_cal','ell_cal'):values=np.array([])
            for q in (0,10,50,90,100):row[f'{field}_p{q}']=float(np.percentile(values,q)) if len(values) else None
        phase_rows.append(row)
        cells=Counter(joint_bin(h,k,cfg) for h,k in zip(frame.h_cal,frame.ell_cal)) if phase!='PROPOSED' else Counter()
        for hi in range(cfg.n_h_bins):
            for ki in range(cfg.n_ell_bins):
                grid_rows.append(dict(phase=phase,h_bin=hi,ell_bin=ki,count=cells[hi,ki],empty=cells[hi,ki]==0,
                    status='NOT_DEFINED_BEFORE_ROOT' if phase=='PROPOSED' else 'AVAILABLE'))
    pd.DataFrame(phase_rows).to_csv(output/'coverage_phase_summary.csv',index=False)
    pd.DataFrame(grid_rows).to_csv(output/'coverage_bin_gaps.csv',index=False)
    stats['phase_summary']=phase_rows
    paths={}
    for label,payload in [('bank',bank),('schedule',schedule)]:
        path=output/f'selected_{label}.pt';atomic_torch_save(path,payload);paths[label]=dict(file=path.name,sha256=file_hash(path))
    atomic_write_json(final,dict(identity=identity,stats=stats,**paths))
    return bank,schedule,stats
