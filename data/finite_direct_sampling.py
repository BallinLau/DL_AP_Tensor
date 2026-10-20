"""Parent-level Beta/empirical sampling with immutable source provenance.

Only data construction lives here. No models, optimizer, cluster calls or
economic equations are changed. Empirical CDF sampling is discrete, with no
interpolation, jitter or kernel smoothing.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import torch

from config import Config
from data.simulate_ts import SimulateTS
from data.simulate_ts_parallel import _initialize_batched_state
from analysis.nested_equilibrium_utils import atomic_torch_save,atomic_write_json,atomic_write_csv,stable_json_hash,tensor_hash

VERSION='finite_direct_beta_empirical_50_v2'
SOURCE_KIND='RECORDED_INITIAL_PARENT_DISTRIBUTION'


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def state_hash(state):
    return tensor_hash(sorted((k,v) for k,v in state.items() if torch.is_tensor(v)))


def capital_groups(capital):
    capital=np.asarray(capital)
    rank=np.empty(len(capital),dtype=np.int64)
    rank[np.argsort(capital,kind='stable')]=np.arange(len(capital))
    return np.minimum(3,rank*4//max(len(capital),1))


def discrete_inverse_cdf(values,uniforms):
    values=np.sort(np.asarray(values,dtype=np.float64))
    if not len(values): raise ValueError('empty empirical CDF')
    u=np.asarray(uniforms,dtype=np.float64)
    if not np.isfinite(u).all() or (u<0).any() or (u>=1).any(): raise ValueError('CDF uniforms outside [0,1)')
    return values[np.floor(u*len(values)).astype(np.int64)]


def template_fingerprint(template):
    return stable_json_hash({k:v for k,v in template.items() if k!='distribution_fingerprint'})


def build_template_bank(source_run,*,expected_source_commit,expected_firm_hash):
    """Read actual saved t0 inputs of a completed finite-direct simulation.

    This intentionally does not claim post-transition/on-policy snapshots:
    699894 retained aggregate dynamic outputs, not dynamic micro states.
    Failed future children never participate in source-template selection.
    """
    source_run=Path(source_run)
    meta=json.loads((source_run/'metadata.json').read_text())
    summary=json.loads((source_run/'summary.json').read_text())
    if summary.get('status')!='COMPLETED' or summary.get('SIMULATION_ROUNDS_COMPLETED')!=9:
        raise RuntimeError('source experiment is not verified complete')
    if meta['source_commit']!=expected_source_commit: raise RuntimeError('empirical source commit mismatch')
    if meta['initial_hashes']['firm']!=expected_firm_hash: raise RuntimeError('empirical source Firm mismatch')
    if meta['representation']!='FINITE_FIRM_DIRECT_AGGREGATION': raise RuntimeError('source is not finite direct')
    templates=[]; manifest=[]; source_hashes={}
    for split in ('train','validation'):
        path=source_run/'banks'/f'{split}.pt'; source_hashes[split]=file_hash(path)
        payload=torch.load(path,map_location='cpu',weights_only=False); state=payload['state']
        for j,path_id in enumerate(state['transition_path_id'].tolist()):
            mask=state['alive'][j].bool() & (state['K'][j]>0)
            b=state['b'][j][mask].numpy().astype(float); k=state['K'][j][mask].numpy().astype(float)
            if not len(b): continue
            if not np.isfinite(b).all() or not np.isfinite(k).all() or (b<0).any() or (b>1).any():
                raise RuntimeError('invalid recorded pre-decision source')
            group=capital_groups(k)
            tid=f'{split}:path{int(path_id)}:t0:pre'
            t=dict(template_id=tid,split=split,source_run=str(source_run),source_commit=meta['source_commit'],
                source_checkpoint_hash=meta['checkpoint_sha256'],source_firm_hash=expected_firm_hash,
                source_bank_sha256=source_hashes[split],source_path=int(path_id),source_time=0,
                pre_or_post_decision='PRE_DECISION',source_kind=SOURCE_KIND,
                firm_count=len(b),b_sorted=np.sort(b).tolist(),
                b_by_capital_quartile=[np.sort(b[group==q]).tolist() for q in range(4)],
                b_mean=float(b.mean()),b_std=float(b.std()),b_support=[float(b.min()),float(b.max())],
                b_quantiles=np.quantile(b,[0,.1,.25,.5,.75,.9,1]).tolist(),
                capital_weighted_b_mean=float(np.sum(k*b)/k.sum()),
                source_total_capital=float(k.sum()),source_x=float(state['x'][j]),
                source_hatc_belief=float(state['hatcf'][j]),source_lnk_belief=float(state['lnkf'][j]))
            t['b_distribution_fingerprint']=stable_json_hash(dict(
                b_sorted=t['b_sorted'],b_by_capital_quartile=t['b_by_capital_quartile']))
            t['distribution_fingerprint']=template_fingerprint(t); templates.append(t)
            manifest.append({key:value for key,value in t.items() if key not in ('b_sorted','b_by_capital_quartile')})
    bank=dict(version=VERSION,source_kind=SOURCE_KIND,templates=templates,source_bank_hashes=source_hashes,
              source_metadata=meta,
              limitation='Templates are actual saved initialization records used by the completed run, not realized post-transition b distributions. Dynamic micro snapshots were not saved; no distributions are reconstructed from aggregate summaries.')
    bank['fingerprint']=stable_json_hash([t['distribution_fingerprint'] for t in templates])
    validate_template_splits(bank)
    return bank,pd.DataFrame(manifest)


def validate_template_splits(bank):
    for t in bank['templates']:
        if template_fingerprint(t)!=t['distribution_fingerprint']: raise RuntimeError('corrupt empirical template')
    sets={s:{(t['source_run'],t['source_path']) for t in bank['templates'] if t['split']==s} for s in ('train','validation')}
    if sets['train']&sets['validation']: raise RuntimeError('train/validation empirical source-path leakage')
    ids=[t['template_id'] for t in bank['templates']]
    if len(set(ids))!=len(ids): raise RuntimeError('duplicate template ID')
    if stable_json_hash([t['distribution_fingerprint'] for t in bank['templates']])!=bank['fingerprint']:
        raise RuntimeError('template bank fingerprint mismatch')
    return dict(source_path_overlap=0,template_id_overlap=0,
                train_templates=sum(t['split']=='train' for t in bank['templates']),
                validation_templates=sum(t['split']=='validation' for t in bank['templates']))


@dataclass(frozen=True)
class SamplingConfig:
    train_parents:int=4096
    validation_parents:int=1024
    firms:int=200
    seed:int=20261020
    alpha_min:float=1.8
    alpha_max:float=3.2
    beta_min:float=4.8
    beta_max:float=7.8
    capital_min:float=45.
    capital_max:float=85.
    conditional_min_count:int=8
    empirical_jitter:float=0.
    def validate(self):
        if self.train_parents%2 or self.validation_parents%2: raise ValueError('50:50 requires even parent counts')
        if self.train_parents<2 or self.validation_parents<2 or self.firms<2: raise ValueError('empty sampling design')
        if self.empirical_jitter!=0: raise ValueError('jitter/smoothing is not authorized')
        if self.conditional_min_count<1: raise ValueError('conditional sample threshold must be positive')
        if (self.capital_min,self.capital_max)!=(45.,85.): raise ValueError('production capital support changed')
        if (self.alpha_min,self.alpha_max,self.beta_min,self.beta_max)!=(1.8,3.2,4.8,7.8):
            raise ValueError('restricted Beta support changed')


def fixed_schedule(config,template_bank,split):
    config.validate(); validate_template_splits(template_bank)
    if split not in ('train','validation'): raise ValueError('unknown split')
    n=config.train_parents if split=='train' else config.validation_parents
    seed=config.seed+(100000 if split=='validation' else 0)
    rng=np.random.default_rng(seed)
    source=np.array(['BETA']*(n//2)+['EMPIRICAL']*(n//2)); source=source[rng.permutation(n)]
    sobol=torch.quasirandom.SobolEngine(6,scramble=True,seed=seed).draw(n).numpy()
    available=sorted(t['template_id'] for t in template_bank['templates'] if t['split']==split)
    if not available: raise RuntimeError(f'no empirical templates for {split}')
    order=np.asarray(available)[rng.permutation(len(available))]; e=0; rows=[]
    for j in range(n):
        kind=str(source[j]); tid=str(order[e%len(order)]) if kind=='EMPIRICAL' else None
        e+=int(kind=='EMPIRICAL')
        parent=j+(2000000 if split=='train' else 3000000)
        rows.append(dict(parent_id=parent,split=split,sampling_source=kind,
            beta_alpha=float(1.8+1.4*sobol[j,0]) if kind=='BETA' else None,
            beta_beta=float(4.8+3.*sobol[j,1]) if kind=='BETA' else None,
            empirical_template_id=tid,initial_state_seed=int(seed+97*j),
            capital_total=float(45.+40.*sobol[j,2]),capital_power=float(.5+1.5*sobol[j,3]),
            beta_rank_mode=('positive' if sobol[j,4]<1/3 else 'negative' if sobol[j,4]>2/3 else 'independent'),
            child_shock_keys=[dict(transition_seed=seed+11,entry_seed=seed+29,path_id=parent,
                                  economic_time=1,branch_id=b) for b in (0,1)]))
    return dict(version=VERSION,split=split,seed=seed,config=asdict(config),
                template_bank_fingerprint=template_bank['fingerprint'],parents=rows,
                fingerprint=stable_json_hash(rows))


def sample_b(spec,capital,template,config):
    rng=np.random.default_rng(spec['initial_state_seed']); n=len(capital)
    if spec['sampling_source']=='BETA':
        b=rng.beta(spec['beta_alpha'],spec['beta_beta'],size=n)
        if spec['beta_rank_mode']=='positive': b[np.argsort(capital,kind='stable')]=np.sort(b)
        elif spec['beta_rank_mode']=='negative': b[np.argsort(capital,kind='stable')]=np.sort(b)[::-1]
        return b,0
    if template is None or template['template_id']!=spec['empirical_template_id']: raise RuntimeError('wrong empirical template')
    if template['split']!=spec['split']: raise RuntimeError('empirical template split leakage')
    group=capital_groups(capital); u=rng.random(n); b=np.empty(n); fallback=0
    for q in range(4):
        mask=group==q; values=template['b_by_capital_quartile'][q]
        if len(values)<config.conditional_min_count:
            values=template['b_sorted']; fallback+=int(mask.sum())
        b[mask]=discrete_inverse_cdf(values,u[mask])
    return b,fallback


def construct_parent_bank(config,template_bank,schedule):
    config.validate()
    if schedule['config']!=asdict(config): raise RuntimeError('sampling configuration changed')
    if schedule!=fixed_schedule(config,template_bank,schedule['split']):
        raise RuntimeError('sampling plan is not the locked schedule')
    if schedule['fingerprint']!=stable_json_hash(schedule['parents']): raise RuntimeError('sampling schedule changed')
    if schedule['template_bank_fingerprint']!=template_bank['fingerprint']: raise RuntimeError('template bank changed')
    n=len(schedule['parents']); seed=schedule['seed']
    # Local RNG construction leaves the caller's global stream unchanged.
    rng_state=torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        sim=SimulateTS(models={'policy_value':None,'sdf_fc1':None},config=Config,n_paths=n,
                       group_size=config.firms,horizon=1,device=torch.device('cpu'),market_clearing_mode='disabled',
                       entry_mode='legacy',enable_entry=False,
                       consumption_aggregation_mode='raw',consumption_floor=0.)
        # Initialization consumes no model. Select the production initial-cohort
        # convention only after construction, without invoking entry or policy.
        sim.entry_mode='investment_compare'
        state=_initialize_batched_state(sim,config.firms)
    finally: torch.set_rng_state(rng_state)
    lookup={t['template_id']:t for t in template_bank['templates']}; rows=[]
    for j,spec in enumerate(schedule['parents']):
        rng=np.random.default_rng(spec['initial_state_seed']+71)
        w=rng.random(config.firms)**spec['capital_power']; w/=w.sum(); capital=spec['capital_total']*w
        template=lookup.get(spec['empirical_template_id']); b,fallback=sample_b(spec,capital,template,config)
        if not np.isfinite(b).all() or (b<0).any() or (b>1).any(): raise RuntimeError('b outside production support')
        state['b'][j]=torch.tensor(b,dtype=torch.float32); state['K'][j]=torch.tensor(capital,dtype=torch.float32)
        state['transition_path_id'][j]=spec['parent_id']
        row=dict(**spec,b_mean=float(b.mean()),b_std=float(b.std()),
            b_p10=float(np.quantile(b,.1)),b_p50=float(np.quantile(b,.5)),b_p90=float(np.quantile(b,.9)),
            high_b_fraction=float((b>.5).mean()),capital_weighted_b_mean=float(np.sum(b*capital)/capital.sum()),
            capital_mean=float(capital.mean()),capital_concentration=float((w*w).sum()),
            b_capital_correlation=float(np.corrcoef(b,capital)[0,1]),conditional_fallback_firms=fallback,
            empirical_source_path=template['source_path'] if template else None,
            empirical_source_time=template['source_time'] if template else None,
            empirical_source_kind=template['source_kind'] if template else None,
            distribution_fingerprint=stable_json_hash(dict(b=b.tolist(),capital=capital.tolist())))
        rows.append(row)
    state['lnkf']=torch.log(state['K'].sum(1)); state['K_birth']=state['K'].clone(); state['bp']=state['b'].clone()
    return state,pd.DataFrame(rows)


def build_or_load_sampling(output,config,template_bank,template_manifest):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    signature=stable_json_hash(dict(version=VERSION,config=asdict(config),template_bank=template_bank['fingerprint']))
    signature_path=output/'sampling_signature.json'
    if signature_path.exists():
        saved=json.loads(signature_path.read_text())
        if saved['fingerprint']!=signature: raise RuntimeError('sampling provenance mismatch; new run required')
        for name,digest in saved['artifact_sha256'].items():
            if file_hash(output/name)!=digest: raise RuntimeError(f'committed sampling artifact corruption: {name}')
        loaded=torch.load(output/'empirical_template_bank.pt',map_location='cpu',weights_only=False)
        if loaded['fingerprint']!=template_bank['fingerprint']: raise RuntimeError('saved template bank mismatch')
        validate_template_splits(loaded)
        banks={}
        for split in ('train','validation'):
            schedule=json.loads((output/f'{split}_parent_sampling_schedule.json').read_text())
            expected=fixed_schedule(config,loaded,split)
            if schedule!=expected: raise RuntimeError('saved parent schedule mismatch')
            p=torch.load(output/f'{split}_initial_bank.pt',map_location='cpu',weights_only=False)
            if state_hash(p['state'])!=p['fingerprint']: raise RuntimeError('saved initial bank corruption')
            banks[split]=p
        manifest=pd.read_csv(output/'macro_initial_distribution_manifest.csv')
        if manifest.distribution_fingerprint.duplicated().any(): raise RuntimeError('duplicate sampled full distribution')
        summary=json.loads((output/'sampling_coverage_summary.json').read_text())
        for split,payload in banks.items():
            if payload['fingerprint']!=summary['splits'][split]['parent_bank_fingerprint']:
                raise RuntimeError('bank does not match committed coverage summary')
        return banks,summary
    # No old cache may be accepted without the full sampling signature.
    if list(output.iterdir()): raise RuntimeError('unversioned sampling output exists; refuse overwrite')
    atomic_torch_save(output/'empirical_template_bank.pt',template_bank)
    atomic_write_csv(output/'empirical_template_manifest.csv',template_manifest)
    banks={}; manifests=[]; summary=dict(version=VERSION,template_leakage_audit=validate_template_splits(template_bank),
        source_kind=SOURCE_KIND,jitter=0.,b_z_dependence_preserved=False,b_eta_dependence_preserved=False,
        b_K_dependence='conditional empirical CDF by new capital quartile; original capital never copied',
        source_limitation=template_bank['limitation'],splits={})
    for split in ('train','validation'):
        schedule=fixed_schedule(config,template_bank,split)
        state,manifest=construct_parent_bank(config,template_bank,schedule)
        payload=dict(state=state,fingerprint=state_hash(state),seed=schedule['seed'],schedule_fingerprint=schedule['fingerprint'])
        atomic_torch_save(output/f'{split}_initial_bank.pt',payload)
        atomic_write_json(output/f'{split}_parent_sampling_schedule.json',schedule)
        banks[split]=payload; manifests.append(manifest)
        empirical=manifest[manifest.sampling_source=='EMPIRICAL']; beta=manifest[manifest.sampling_source=='BETA']
        template_lookup={t['template_id']:t for t in template_bank['templates']}
        selected_shapes={template_lookup[tid].get('b_distribution_fingerprint',
                          stable_json_hash(template_lookup[tid]['b_sorted']))
                         for tid in empirical.empirical_template_id}
        summary['splits'][split]=dict(parents=len(manifest),beta_parents=len(beta),empirical_parents=len(empirical),
            unique_empirical_templates=int(empirical.empirical_template_id.nunique()),
            unique_selected_empirical_CDF_shapes=len(selected_shapes),
            template_reuse_fraction=float(1-empirical.empirical_template_id.nunique()/len(empirical)),
            conditional_fallback_firms=int(empirical.conditional_fallback_firms.sum()),
            parent_bank_fingerprint=payload['fingerprint'],sampling_schedule_fingerprint=schedule['fingerprint'],
            b_mean_min=float(manifest.b_mean.min()),b_mean_max=float(manifest.b_mean.max()))
    atomic_write_csv(output/'macro_initial_distribution_manifest.csv',pd.concat(manifests,ignore_index=True))
    atomic_write_json(output/'sampling_coverage_summary.json',summary)
    artifacts={p.name:file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}
    atomic_write_json(signature_path,dict(version=VERSION,fingerprint=signature,config=asdict(config),
                                         artifact_sha256=artifacts))
    return banks,summary
