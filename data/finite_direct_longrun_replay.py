"""Train-only historical FC1 replay. Never an SDF pair source.

Selection unit is a unique parent-time group, not a duplicated child row.
Fixed edges are locked before D1; Validation never fits selection edges.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis.nested_equilibrium_utils import atomic_write_json, stable_json_hash
from data.finite_direct_sampling import file_hash
from data.finite_direct_macro_contract import select_training_rows
from types import SimpleNamespace

VERSION='HISTORICAL_POLICY_MACRO_REPLAY_V1'
FIELDS=['x_t','x_t1','Hatc_cal_t','LnK_cal_t','Hatc_cal_t1','LnK_cal_t1','C_raw_t1','K_cal_t1']


def legal(frame):
    f=frame.copy()
    for name in FIELDS:
        if name not in f:f[name]=np.nan
    if f.empty:return f
    if set(f.split)!= {'train'}:raise ValueError('Replay/coverage must be Train-only')
    selected=eligibility(f).fc1
    # Explicitly retain only defined ledger and conditional targets.
    return selected.loc[np.isfinite(selected[FIELDS]).all(axis=1) &
        (selected.C_raw_t1>0)&(selected.K_cal_t1>0)].copy()


def eligibility(frame):
    """Same exact ledger gates, vectorized sibling-context and pair indices.

    No per-parent Python group iteration or neural call. Both FC1 and SDF
    indices are obtained from the single checked table.
    """
    f=frame.reset_index(drop=True).copy();keys=['split','parent_candidate_id','t','child_branch']
    for name in FIELDS:
        if name not in f:f[name]=np.nan
    if f.duplicated(keys).any():raise ValueError('duplicate economic child')
    if not f.child_branch.isin([0,1]).all():raise ValueError('invalid child branch')
    if not f.split.isin(['train','validation']).all():raise ValueError('unknown split')
    ids=f[['parent_candidate_id','t']].to_numpy(dtype=float)
    if not np.isfinite(ids).all() or not np.equal(ids,np.floor(ids)).all() or (ids[:,0]<0).any() or (ids[:,1]<1).any():
        raise ValueError('invalid parent/time')
    if set(f.loc[f.split=='train','parent_candidate_id'])&set(f.loc[f.split=='validation','parent_candidate_id']):
        raise ValueError('Train/Validation leakage')
    if not f.economic_state_valid.isin([True,False]).all():raise ValueError('invalid validity flag')
    claimed=f.economic_state_valid.astype(bool)
    ok=np.isfinite(f[FIELDS].to_numpy(dtype=float)).all(1)&(f.C_raw_t1>0)&(f.K_cal_t1>0)
    if (claimed&~ok).any():raise ValueError('claimed valid undefined label')
    selected=f.loc[claimed].copy()
    if not np.allclose(selected.Hatc_cal_t1,np.log(selected.C_raw_t1/selected.K_cal_t1),rtol=0,atol=2e-6):
        raise ValueError('Hatc ledger mismatch')
    if not np.allclose(selected.LnK_cal_t1,np.log(selected.K_cal_t1),rtol=0,atol=2e-6):raise ValueError('LnK ledger mismatch')
    group=['split','parent_candidate_id','t']
    contexts=selected.groupby(group)[['x_t','Hatc_cal_t','LnK_cal_t']].nunique(dropna=False)
    if (contexts.to_numpy()>1).any():raise ValueError('siblings disagree on parent context')
    sizes=selected.groupby(group).child_branch.transform('size')
    f['fc1_eligible']=claimed&ok
    f['sdf_eligible']=False
    f.loc[selected.index,'sdf_eligible']=(sizes==2).to_numpy()
    return SimpleNamespace(fc1=f.loc[f.fc1_eligible].copy(),sdf_pairs=f.loc[f.sdf_eligible].copy(),audit=f)


def commit_pool(stores,output,*,source_outer,model_hashes,initial_design,outer_checkpoint_sha):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'commit.json'
    identity=dict(version=VERSION,source_outer=source_outer,model_hashes=model_hashes,
        initial_design=initial_design,outer_checkpoint_sha256=outer_checkpoint_sha,burnin=20,
        source_manifests=[file_hash(s.root/'manifest.json') for s in stores])
    if path.exists():
        result=json.loads(path.read_text())
        if result['identity']!=identity:raise ValueError('immutable replay pool identity mismatch')
        load_pool(output,consumer_outer=source_outer+1)
        return result
    pieces=[]
    for store in stores:
        if store.manifest['provenance']['branches']!=1:raise ValueError('H100 must be single-spine')
        for f in store.rows():
            f=legal(f)
            if f.empty:continue
            f=f.loc[(f.t-1>=20)&(f.t-1<100)].copy()
            f['parent_time']=f.t-1
            f['initial_mode']=store.manifest['provenance']['initial_mode']
            pieces.append(f)
    f=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=FIELDS+['split','parent_candidate_id','t','child_branch','economic_state_valid','parent_time'])
    if len(f) and f.duplicated(['parent_candidate_id','t']).any():raise ValueError('duplicated replay parent-time')
    f['source_outer']=source_outer;f['replay_semantics']='HISTORICAL_POLICY_MACRO_REPLAY'
    f['original_lineage']=f.parent_candidate_id
    f['valid_label']=True
    for name in ('firm','fc1','sdf'):
        f['source_'+name+'_hash']=model_hashes[name]
    if 'wealth' in model_hashes:f['source_wealth_hash']=model_hashes['wealth']
    data=output/'macro_rows.csv';temporary=output/'macro_rows.csv.tmp'
    f.to_csv(temporary,index=False);temporary.replace(data)
    result=dict(identity=identity,rows_file=data.name,rows_sha256=file_hash(data),
        eligible_parent_times=len(f),unique_lineages=int(f.parent_candidate_id.nunique()),
        status='COMMITTED' if len(f) else 'EMPTY_LONGRUN_POOL',sdf_pair_source=False)
    result['fingerprint']=stable_json_hash(result)
    atomic_write_json(path,result)
    return result


def load_pool(path,*,consumer_outer):
    path=Path(path);m=json.loads((path/'commit.json').read_text())
    fingerprint=m.pop('fingerprint')
    if stable_json_hash(m)!=fingerprint:raise ValueError('replay commit corruption')
    if m['identity']['source_outer']!=consumer_outer-1:raise ValueError('only previous committed Outer replay allowed')
    data=path/m['rows_file']
    if data.parent!=path or file_hash(data)!=m['rows_sha256']:raise ValueError('replay content corruption')
    f=pd.read_csv(data)
    if len(f) and ((f.parent_time<20).any() or (f.parent_time>=100).any()):raise ValueError('burn-in replay forbidden')
    f['replay_age']=consumer_outer-m['identity']['source_outer']
    return f


def fit_edges(train_frames,reference=((-6,2),(-2,5)),bins=8):
    values=[f[['Hatc_cal_t','LnK_cal_t']].to_numpy() for f in train_frames if len(f)]
    data=np.concatenate(values) if values else np.empty((0,2))
    edges=[]
    for axis,(low,high) in enumerate(reference):
        if len(data):low=min(low,float(data[:,axis].min()));high=max(high,float(data[:,axis].max()))
        edges.append(np.linspace(low,high,bins+1).tolist())
    return edges


def cells(frame,edges):
    # Separate directional low/high coordinates, never collapse overflow.
    output=[]
    for e,c in zip(edges,('Hatc_cal_t','LnK_cal_t')):
        values=frame[c].to_numpy();indices=np.searchsorted(e,values,side='right')-1
        output.append(np.where(values==e[-1],len(e)-2,indices))
    return np.stack(output,axis=1)


def select_mixture(coverage,replay,*,edges,seed=20261022,target=4096,lineage_cap_fraction=.05,
                   on_policy_fraction=.5):
    coverage=legal(coverage);replay=legal(replay) if len(replay) else replay.copy()
    if len(replay) and ((replay.t-1<20).any() or (replay.child_branch!=0).any()):
        raise ValueError('invalid historical single-spine replay')
    rng=np.random.default_rng(seed);sources={'LONGRUN':replay,'COVERAGE':coverage}
    groups={};audit={};chosen=[]
    for name,f in sources.items():
        if f.empty:groups[name]=[];continue
        f=f.copy();f['replay_source']=name
        f['original_lineage']=f.get('original_lineage',f.parent_candidate_id)
        f['_hbin'],f['_kbin']=cells(f,edges).T
        f['_band']=np.minimum(2,np.floor((f.t-1)/max(1,(100 if name=='LONGRUN' else 10)/3))).astype(int)
        # Vectorized first-row index creates unique group descriptors. Child
        # siblings are retained together; no synthetic second Child is made.
        descriptors=f.drop_duplicates(['parent_candidate_id','t']).copy()
        descriptors['_random']=rng.random(len(descriptors))
        descriptors['_family']=descriptors.get('b_distribution_family','UNKNOWN')
        descriptors=descriptors.sort_values('_random')
        descriptors['_rank']=descriptors.groupby(['_hbin','_kbin','_band','_family'],dropna=False).cumcount()
        descriptors=descriptors.sort_values(['_rank','_random'])
        groups[name]=[g for g in descriptors.to_dict('records')]
        sources[name]=f
    if not 0<=float(on_policy_fraction)<=1:raise ValueError('on_policy_fraction must be in [0,1]')
    if target<1:raise ValueError('target must be positive')
    long_target=min(int(round(target*float(on_policy_fraction))),len(groups['LONGRUN']))
    requests={'LONGRUN':long_target,'COVERAGE':target-long_target}
    for name in ('LONGRUN','COVERAGE'):
        counts={};bin_counts={};selected=[];limit=max(1,int(np.ceil(target*lineage_cap_fraction)))
        occupied=len({(r['_hbin'],r['_kbin']) for r in groups[name]})
        bin_limit=max(1,int(np.ceil(max(target*.125,requests[name]/max(1,occupied)))))
        for row in groups[name]:
            lineage=row['original_lineage']
            if counts.get(lineage,0)>=limit:continue
            cell=(row['_hbin'],row['_kbin'])
            if bin_counts.get(cell,0)>=bin_limit:continue
            selected.append((row['parent_candidate_id'],row['t']))
            counts[lineage]=counts.get(lineage,0)+1
            bin_counts[cell]=bin_counts.get(cell,0)+1
            if len(selected)>=requests[name]:break
        if requests[name]==0:selected=[]
        f=sources[name]
        if selected:
            idx=pd.MultiIndex.from_tuples(selected,names=['parent_candidate_id','t'])
            piece=f.set_index(['parent_candidate_id','t']).loc[idx].reset_index()
            chosen.append(piece)
        audit[name]=dict(requested=requests[name],eligible=len(groups[name]),selected=len(selected),
            underfilled=requests[name]-len(selected),per_lineage_cap=limit,per_bin_cap=bin_limit)
        if name=='LONGRUN':requests['COVERAGE']=target-len(selected)
    result=pd.concat(chosen,ignore_index=True) if chosen else coverage.iloc[:0].copy()
    if len(result):
        grouped=result.drop_duplicates(['replay_source','parent_candidate_id','t'])
        counts=grouped.groupby(['replay_source','original_lineage']).size().to_numpy()
        occupancy=grouped.groupby(['_hbin','_kbin']).size().to_numpy()
        audit.update(lineage_ESS=float(counts.sum()**2/(counts@counts)),
            largest_lineage_share=float(counts.max()/counts.sum()),occupied_2d_bins=len(occupancy),
            largest_bin_share=float(occupancy.max()/occupancy.sum()),
            bin_entropy=float(-np.sum(occupancy/occupancy.sum()*np.log(occupancy/occupancy.sum()))))
    return result.drop(columns=[c for c in result if c.startswith('_')]),audit
