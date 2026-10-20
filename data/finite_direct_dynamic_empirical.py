"""Versioned Train-only realized-b CDFs. CPU storage, no optimization.

Parent-cluster sampling: uniform cluster -> uniform available sibling.
Only b is resampled; original capital/z/eta/i/firm IDs/CRN remain unchanged.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from analysis.nested_equilibrium_utils import atomic_torch_save,atomic_write_json,stable_json_hash
from data.finite_direct_sampling import capital_groups,file_hash,state_hash

VERSION='train_dynamic_realized_b_parent_cluster_v1'
CONFIG=dict(version=VERSION,recent_count=2867,archive_count=1229,empirical_count=4096,
    seed_base=202610214,seed_rule='seed_base + target_outer*100003 + parent_id*97',
    beta_empirical='50:50',recent_archive='70:30',conditional_min_count=8,
    source='POST_SDF_TRAIN_EMPIRICAL_SOURCE',cluster='uniform_source_parent_lineage_then_uniform_available_snapshot_child',
    initialization='LOCKED_JOB699894_T0',validation='FIXED_INITIAL_BANK',
    missing_recent='HARD_FAILURE_NO_SUBSTITUTION',last_bank_outer=11)


def fingerprint(value):
    return stable_json_hash({k:v for k,v in value.items() if k!='fingerprint'})


def checked(value):
    if value.get('fingerprint')!=fingerprint(value): raise ValueError('pool/schedule fingerprint corruption')
    if 'recent' in value and (value.get('version')!=VERSION or value.get('config')!=CONFIG):
        raise ValueError('dynamic algorithm/config mismatch')
    return value


def template(b,k,*,identity,source):
    b=np.asarray(b,dtype=float);k=np.asarray(k,dtype=float)
    if len(b)==0 or b.shape!=k.shape or not np.isfinite(b).all() or not np.isfinite(k).all() or (k<=0).any() or (b<0).any() or (b>1).any():
        raise ValueError('INVALID_PRE_DECISION_B_K')
    q=capital_groups(k)
    corr=float(np.corrcoef(b,k)[0,1]) if b.std()>0 and k.std()>0 else None
    return dict(template_id=identity,**source,b_sorted=np.sort(b).tolist(),
        b_by_capital_quartile=[np.sort(b[q==j]).tolist() for j in range(4)],
        firm_count=len(b),b_mean=float(b.mean()),b_std=float(b.std()),b_variance=float(b.var()),
        b_p10=float(np.quantile(b,.1)),b_p50=float(np.quantile(b,.5)),b_p90=float(np.quantile(b,.9)),
        high_b_fraction=float((b>.5).mean()),capital_weighted_b_mean=float((b*k).sum()/k.sum()),
        b_capital_correlation=corr,capital_sum=float(k.sum()),capital_quantiles=np.quantile(k,[0,.25,.5,.75,1]).tolist())


def capture_templates(state,*,split,outer,branch,model_hashes):
    if split!='train': raise ValueError('Validation cannot enter Train empirical pool')
    # Batch-transfer only b/K/identity/mask: no graph, no 7D state history.
    b=state['b'].detach().cpu().numpy();k=state['K'].detach().cpu().numpy()
    alive=state['alive'].detach().cpu().numpy();ids=state['transition_path_id'].cpu().tolist()
    previous=state['transition_previous_alive'].detach().cpu().numpy()
    old_exit_counts=(previous & ~alive[:,:previous.shape[1]]).sum(axis=1)
    result=[]
    for j,pid in enumerate(ids):
        source=dict(split='train',source_outer=outer,source_time=1,source_branch=branch,
            source_path=int(pid),cluster_id=f'continuation_train:parent{pid}',model_hashes=model_hashes,
            source_stage=CONFIG['source'],source_kind='REACHABLE_PRE_DECISION_DISTRIBUTION',
            parent_default_count=int(old_exit_counts[j]),
            post_decision_accounting_valid='NOT_YET_EVALUATED')
        try:
            t=template(b[j,alive[j]],k[j,alive[j]],identity=f'outer{outer}:parent{pid}:branch{branch}',source=source)
            result.append(dict(dynamic_template=t,split=split,source_time=1))
        except ValueError as e:
            result.append(dict(template_failure=dict(**source,reason=str(e)),split=split,source_time=1))
    return result


def initial_pool(template_bank):
    templates=[]
    for old in template_bank['templates']:
        if old['split']!='train': continue
        if old.get('source_time')!=0: raise ValueError('initial pool must honestly identify locked t0')
        t=deepcopy(old);t.update(source_outer=-1,source_stage='JOB699894_T0',source_kind='RECORDED_INITIAL_PARENT_DISTRIBUTION',
            cluster_id=f'job699894:parent{old["source_path"]}',source_branch=-1,model_hashes={'firm':old.get('source_firm_hash')})
        templates.append(t)
    if not templates: raise ValueError('no initial Train templates')
    pool=dict(version=VERSION,target_outer=4,recent=[],archive=templates,source_outer=None,
        initialization='LOCKED_JOB699894_T0',config=CONFIG)
    pool['fingerprint']=fingerprint(pool);return pool


def next_pool(previous,captures,*,source_outer,source_hashes,source_manifest_sha):
    checked(previous)
    if previous['target_outer']!=source_outer: raise ValueError('wrong source outer/pool')
    recent=[];failures=[]
    for capture in captures:
        if capture['split']!='train': raise ValueError('Validation leakage')
        if 'template_failure' in capture: failures.append(capture['template_failure']);continue
        if 'dynamic_template' not in capture: continue
        t=deepcopy(capture['dynamic_template'])
        if t['split']!='train' or t['source_outer']!=source_outer or t['source_time']!=1 or t['source_branch'] not in (0,1) or t['source_kind']!='REACHABLE_PRE_DECISION_DISTRIBUTION' or t['source_stage']!=CONFIG['source'] or t['model_hashes']!=source_hashes:
            raise ValueError('wrong source timing/stage/model')
        recent.append(t)
    keys=[t['template_id'] for t in recent]
    if len(keys)!=len(set(keys)): raise ValueError('duplicate template')
    sibling_keys=[(t['cluster_id'],t['source_branch']) for t in recent]
    if len(sibling_keys)!=len(set(sibling_keys)): raise ValueError('duplicate Parent/sibling branch')
    if not recent: raise ValueError('NO_RECENT_TEMPLATES_NO_OLD_POOL_SUBSTITUTION')
    archive=previous['archive']+previous['recent']
    if any(t['split']!='train' or t['source_outer']>=source_outer for t in archive): raise ValueError('future/validation archive')
    result=dict(version=VERSION,target_outer=source_outer+1,recent=recent,archive=archive,source_outer=source_outer,
        source_model_hashes=source_hashes,source_manifest_sha=source_manifest_sha,failures=failures,config=CONFIG)
    result['fingerprint']=fingerprint(result);return result


def resample_train(initial_bank,initial_schedule,pool,*,outer,empirical_count=4096):
    checked(pool)
    if pool['target_outer']!=outer: raise ValueError('uncommitted/wrong pool version')
    bank=deepcopy(initial_bank);schedule=deepcopy(initial_schedule)
    specs=[r for r in schedule['parents'] if r['sampling_source']=='EMPIRICAL']
    if len(specs)!=empirical_count: raise ValueError('wrong empirical capacity')
    if sum(r['sampling_source']=='BETA' for r in schedule['parents'])!=empirical_count:
        raise ValueError('Parent mixture must be exactly 50:50')
    rows_by_id={r['parent_id']:j for j,r in enumerate(schedule['parents'])}
    seed=CONFIG['seed_base']+outer*100003;g=np.random.default_rng(seed)
    if outer==4:
        # Initial reviewed bank remains bitwise identical, including its t0 b.
        specs=[]
    elif not pool['recent']: raise ValueError('no Recent pool')
    recent_count=int(np.floor(empirical_count*.7))
    sources=np.array(['RECENT']*recent_count+['ARCHIVE']*(empirical_count-recent_count));g.shuffle(sources)
    clusters={}
    for name,values in [('RECENT',pool['recent']),('ARCHIVE',pool['archive'])]:
        groups={}
        for t in values:
            if t['split']!='train': raise ValueError('Validation leakage')
            groups.setdefault(t['cluster_id'],[]).append(t)
        clusters[name]=[(key,sorted(v,key=lambda t:t['template_id'])) for key,v in sorted(groups.items())]
    history=[]
    for i,spec in enumerate(specs):
        choices=clusters[sources[i]]
        if not choices: raise ValueError('empty declared pool; no silent fallback')
        pid=spec['parent_id'];rng=np.random.default_rng(seed+pid*97)
        key,children=choices[int(rng.integers(len(choices)))];t=children[int(rng.integers(len(children)))]
        j=rows_by_id[pid];capital=bank['state']['K'][j].numpy();q=capital_groups(capital);u=rng.random(len(capital));b=np.empty_like(capital);fallback=0
        for quartile in range(4):
            mask=q==quartile;values=t['b_by_capital_quartile'][quartile]
            if len(values)<CONFIG['conditional_min_count']: values=t['b_sorted'];fallback+=int(mask.sum())
            # Template CDFs are sorted once at extraction, not per resampled
            # Parent/firm. Discrete inverse CDF, no interpolation or jitter.
            values=np.asarray(values,dtype=float)
            b[mask]=values[np.floor(u[mask]*len(values)).astype(np.int64)]
        bank['state']['b'][j]=torch.from_numpy(b);bank['state']['bp'][j]=torch.from_numpy(b.copy())
        spec.update(empirical_template_id=t['template_id'],template_version=pool['fingerprint'],
            sampling_seed=seed+pid*97,source_cluster=key,source_branch=t['source_branch'],pool_source=str(sources[i]))
        stats=template(b,capital,identity=t['template_id'],source={})
        history.append(dict(outer=outer,parent_id=pid,pool_source=str(sources[i]),source_outer=t['source_outer'],
            template_id=t['template_id'],cluster_id=key,fallback_fraction=fallback/len(capital),
            **{k:v for k,v in stats.items() if k not in ('template_id','b_sorted','b_by_capital_quartile')}))
    schedule.update(outer_id=outer,empirical_pool_fingerprint=pool['fingerprint'],bank_version=f'outer_{outer:03d}')
    schedule['fingerprint']=fingerprint(schedule)
    bank.update(fingerprint=state_hash(bank['state']),schedule_fingerprint=schedule['fingerprint'],
        empirical_pool_fingerprint=pool['fingerprint'],bank_version=f'outer_{outer:03d}')
    return bank,schedule,history


def save_bundle(root,pool,bank,schedule,history):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    # Content-addressed payloads prevent uncommitted retries from overwriting a
    # file referenced by a prior StageJournal checkpoint.
    from analysis.finite_direct_continuation_state import state_digest
    names={}
    for label,value in [('pool',pool),('bank',bank),('schedule',schedule)]:
        name=f'{label}_{state_digest(value)}.pt';path=root/name
        if path.exists():
            from analysis.finite_direct_continuation_state import safe_load
            if state_digest(safe_load(path))!=state_digest(value): raise ValueError('immutable file corruption')
        else: atomic_torch_save(path,value)
        names[label]=dict(path=str(path),sha256=file_hash(path))
    names.update(outer=pool['target_outer'],pool_fingerprint=pool['fingerprint'],bank_fingerprint=bank['fingerprint'],history=history)
    names['crn_manifest_hash']=stable_json_hash([r['child_shock_keys'] for r in schedule['parents']])
    atomic_write_json(root/f'bundle_{pool["fingerprint"]}.json',names)
    return names


def source_summary(templates):
    """Uniform Parent cluster, uniform sibling, uniform firm within snapshot."""
    counts={}
    for t in templates:counts[t['cluster_id']]=counts.get(t['cluster_id'],0)+1
    weights=np.array([1/len(counts)/counts[t['cluster_id']] for t in templates])
    values=np.concatenate([np.asarray(t['b_sorted'],dtype=float) for t in templates])
    point_weights=np.concatenate([np.full(len(t['b_sorted']),w/len(t['b_sorted'])) for t,w in zip(templates,weights)])
    order=np.argsort(values);v=values[order];w=point_weights[order];cdf=np.cumsum(w)
    quant=lambda q:float(v[min(np.searchsorted(cdf,q),len(v)-1)])
    mean=float((values*point_weights).sum())
    def average(key):
        available=[(t[key],w) for t,w in zip(templates,weights) if t.get(key) is not None]
        return float(sum(v*w for v,w in available)/sum(w for _,w in available)) if available else None
    return dict(source_weighting='UNIFORM_PARENT_THEN_SIBLING_THEN_FIRM',b_mean=mean,
        b_std=float(np.sqrt(((values-mean)**2*point_weights).sum())),b_p10=quant(.1),b_p50=quant(.5),b_p90=quant(.9),
        high_b_fraction=float(((values>.5)*point_weights).sum()),
        capital_weighted_b_mean=average('capital_weighted_b_mean'),b_capital_correlation=average('b_capital_correlation'),
        cross_template_statistics='capital-weighted b and correlation are averages of snapshot statistics',
        source_surviving_firm_count=average('firm_count'),source_parent_default_count=average('parent_default_count'))


def load_bundle(record):
    from analysis.finite_direct_continuation_state import safe_load
    result=[]
    for label in ('pool','bank','schedule'):
        info=record[label]
        if file_hash(info['path'])!=info['sha256']: raise ValueError('committed dynamic artifact SHA mismatch')
        result.append(safe_load(info['path']))
    pool,bank,schedule=result;checked(pool);checked(schedule)
    if state_hash(bank['state'])!=bank['fingerprint'] or bank['empirical_pool_fingerprint']!=pool['fingerprint'] or schedule['empirical_pool_fingerprint']!=pool['fingerprint'] or bank['schedule_fingerprint']!=schedule['fingerprint']:
        raise ValueError('dynamic pool/bank/schedule mismatch')
    if pool['target_outer']!=record['outer'] or bank['bank_version']!=f'outer_{record["outer"]:03d}': raise ValueError('wrong bank version')
    if pool['fingerprint']!=record['pool_fingerprint'] or bank['fingerprint']!=record['bank_fingerprint']:
        raise ValueError('committed bank identity differs from record')
    crn=stable_json_hash([r['child_shock_keys'] for r in schedule['parents']])
    if crn!=record['crn_manifest_hash']: raise ValueError('CRN mismatch')
    return pool,bank,schedule
