"""Lazy committed H50 template sources: lineage -> time -> sibling -> CDF.

No validation data, future accounting filter, added noise or changed capital.
Archive represents committed earlier H50 source stores, never uncommitted rows.
"""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from analysis.nested_equilibrium_utils import stable_json_hash
from data.finite_direct_long_horizon_runtime import ShardStore
from data.finite_direct_sampling import file_hash,capital_groups,state_hash


def resample_from_sources(initial_bank,schedule,sources,*,outer,initial_archive):
    if not sources:raise ValueError('no committed H50 empirical source')
    indices=[]
    for source in sources:
        root=Path(source['path'])
        if file_hash(root/'manifest.json')!=source['manifest_sha256']:
            raise ValueError('empirical source manifest corruption')
        if source['outer']>=outer:raise ValueError('future empirical source')
        store=ShardStore(root)
        if store.manifest['provenance']['stage']!='POST_SDF_TRAIN_EMPIRICAL_SOURCE' or store.manifest['provenance']['split']!='train':
            raise ValueError('wrong empirical stage')
        grouped=defaultdict(lambda:defaultdict(list))
        for shard_index,shard in enumerate(store.chunks()):
            for item_index,capture in enumerate(shard['templates']):
                if capture['split']!='train':raise ValueError('Validation leakage')
                t=capture.get('dynamic_template')
                if t is not None:
                    grouped[t['cluster_id']][t['source_time']].append((shard_index,item_index))
        indices.append((store,grouped))
    # For the first new update no prior H50 archive exists. The locked initial
    # empirical source supplies Archive; record that explicitly, not fallback.
    bank=deepcopy(initial_bank);rng=np.random.default_rng(20261021+outer*100003)
    specs=[(j,s) for j,s in enumerate(schedule['parents']) if s['sampling_source']=='EMPIRICAL']
    if len(specs)!=4096 or sum(s['sampling_source']=='BETA' for s in schedule['parents'])!=4096:
        raise ValueError('8192 Parents and exact 50:50 required')
    pool_flags=np.array(['RECENT']*2867+['ARCHIVE']*1229);rng.shuffle(pool_flags)
    chosen=defaultdict(list);history=[]
    if not initial_archive or any(t['split']!='train' for t in initial_archive):
        raise ValueError('locked Train archive required; no silent fallback')
    initial_groups=defaultdict(lambda:defaultdict(list))
    for index,t in enumerate(initial_archive):
        key=t.get('cluster_id',f'INITIAL:{t.get("source_path",t["template_id"])}')
        initial_groups[key][t.get('source_time',0)].append((0,index))
    def draw_into(j,t):
        k=bank['state']['K'][j].numpy();q=capital_groups(k);b=np.empty_like(k);u=rng.random(len(k))
        for quartile in range(4):
            mask=q==quartile;values=t['b_by_capital_quartile'][quartile]
            if len(values)<8:values=t['b_sorted']
            values=np.asarray(values);b[mask]=values[np.floor(u[mask]*len(values)).astype(int)]
        bank['state']['b'][j]=torch.from_numpy(b);bank['state']['bp'][j]=torch.from_numpy(b.copy())
    for (j,spec),flag in zip(specs,pool_flags):
        available=[indices[-1]] if flag=='RECENT' else indices[:-1]+[(None,initial_groups)]
        # Uniform original lineage over the union, not uniform snapshot count.
        lineages=sorted({pid for _,groups in available for pid in groups})
        if not lineages:raise ValueError('NO_RECENT_H50_TEMPLATES')
        pid=lineages[int(rng.integers(len(lineages)))];options=[(store,g[pid]) for store,g in available if pid in g]
        all_times=sorted({time for _,times in options for time in times})
        time=all_times[int(rng.integers(len(all_times)))]
        snapshots=[(store,shard,item) for store,times in options for shard,item in times.get(time,[])]
        store,shard,item=snapshots[int(rng.integers(len(snapshots)))]
        if store is None:
            t=initial_archive[item];draw_into(j,t)
            history.append(dict(parent_id=spec['parent_id'],pool_source='ARCHIVE',
                source='LOCKED_INITIAL_ARCHIVE',template_id=t['template_id']))
            continue
        chosen[str(store.root)].append((j,shard,item,spec['parent_id'],str(flag)))
    for store,_ in indices:
        selected=defaultdict(list)
        for row in chosen[str(store.root)]:selected[row[1]].append(row)
        for shard_index,shard in enumerate(store.chunks()):
            for j,_,item,pid,flag in selected.get(shard_index,[]):
                t=shard['templates'][item]['dynamic_template'];draw_into(j,t)
                history.append(dict(parent_id=pid,pool_source=flag,template_id=t['template_id'],
                    source_time=t['source_time'],source_branch=t['source_branch'],source_lineage=t['cluster_id']))
    bank.update(fingerprint=state_hash(bank['state']),bank_version=f'H50_outer_{outer}',
        empirical_pool_fingerprint=stable_json_hash(sources),h50_empirical_history=history)
    return bank
