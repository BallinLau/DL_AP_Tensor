"""Deterministic bounded reservoirs, weighted by lineage/time/macro strata.

Storage is lazy; only the approved bounded collocation budget is materialized.
No Bellman targets are persisted or reused between forecast/calculated inputs.
"""
from collections import defaultdict
import math
import numpy as np
import pandas as pd
import torch

from analysis.nested_equilibrium_utils import stable_json_hash
from training.inner_fixed_point import InnerFPCollocationBank
from data.finite_direct_macro_contract import select_training_rows
from data.finite_direct_initial_hstar import VERSION


def time_bin(time):
    return 0 if time < 10 else 1 if time < 30 else 2


def macro_bin(h, k):
    if not math.isfinite(h) or not math.isfinite(k):
        raise ValueError('undefined macro stratum')
    return math.floor(h), math.floor(k)


def bounded_quotas(available, budget):
    """Redistribute unused quota only to distinct available observations."""
    if budget<0:raise ValueError('negative sampling budget')
    quotas={band:0 for band in (0,1,2)}
    remaining=min(budget,sum(available.values()))
    while remaining:
        active=[band for band in quotas if quotas[band]<available.get(band,0)]
        portion=max(1,remaining//len(active))
        for band in active:
            add=min(portion,available[band]-quotas[band],remaining)
            quotas[band]+=add;remaining-=add
    return quotas


def sample_macro_rows(store, *, pair_budget, seed, complete_pairs_only=False):
    """Bounded paired reservoir: equal lineage, then time/state strata.

    Keep complete pairs together when both are eligible. Single legal FC1
    children remain eligible, but SDF will still reject incomplete groups.
    Priorities normalize each encountered lineage/stratum's row multiplicity;
    the resulting reweighting is explicit optimization sampling, not population.
    """
    import heapq
    rng=np.random.default_rng(seed); counts=defaultdict(int); coverage=[]
    def eligible(frame):
        for name in ('x_t','x_t1','Hatc_cal_t','LnK_cal_t','Hatc_cal_t1','LnK_cal_t1','C_raw_t1','K_cal_t1'):
            if name not in frame:frame[name]=np.nan
        return select_training_rows(frame)
    for frame in store.rows():
        selected=eligible(frame)
        coverage.append(dict(attempted=len(frame),fc1=len(selected.fc1),sdf_children=len(selected.sdf_pairs)))
        sampling=selected.sdf_pairs if complete_pairs_only else selected.fc1
        for (split,pid,t),g in sampling.groupby(['split','parent_candidate_id','t']):
            first=g.iloc[0]; key=(int(pid),time_bin(int(t)-1),*macro_bin(first.Hatc_cal_t,first.LnK_cal_t))
            counts[key]+=1
    bands=sorted({k[1] for k in counts});keys_per_lineage=defaultdict(int)
    for pid,band,h,k in counts:keys_per_lineage[pid,band]+=1
    if not bands:raise ValueError('NO_COMPLETE_LONG_HORIZON_SDF_PAIRS' if complete_pairs_only else 'NO_LEGAL_LONG_HORIZON_MACRO_ROWS')
    available={band:sum(count for key,count in counts.items() if key[1]==band) for band in (0,1,2)}
    quotas=bounded_quotas(available,pair_budget)
    heaps={band:[] for band in bands}
    for frame in store.rows():
        selected=eligible(frame);sampling=selected.sdf_pairs if complete_pairs_only else selected.fc1
        for (split,pid,t),g in sampling.groupby(['split','parent_candidate_id','t']):
            first=g.iloc[0];band=time_bin(int(t)-1)
            key=(int(pid),band,*macro_bin(first.Hatc_cal_t,first.LnK_cal_t))
            priority=float(rng.random())**(counts[key]*keys_per_lineage[pid,band])
            entry=(priority,float(rng.random()),g.to_dict('records'))
            heap=heaps[band];quota=quotas[band]
            if len(heap)<quota: heapq.heappush(heap,entry)
            elif quota and entry[:2]>heap[0][:2]: heapq.heapreplace(heap,entry)
    rows=[r for heap in heaps.values() for _,_,records in heap for r in records]
    if not rows: raise ValueError('NO_LEGAL_LONG_HORIZON_MACRO_ROWS')
    result=pd.DataFrame(rows).sort_values(['split','parent_candidate_id','t','child_branch']).reset_index(drop=True)
    return result,dict(pair_budget=pair_budget,seed=seed,source_coverage=coverage,
        index_sha256=stable_json_hash(result[['split','parent_candidate_id','t','child_branch']].to_dict('records')),
        rule='EQUAL_NONEMPTY_TIME_BAND_THEN_WEIGHTED_LINEAGE_MACRO_RESERVOIR',
        time_band_quotas=quotas,available_parent_times_by_band=available,
        requested_parent_time_budget=pair_budget,actual_parent_times=sum(len(h) for h in heaps.values()),
        unused_quota=pair_budget-sum(len(h) for h in heaps.values()),
        actual_unique_child_rows=len(result),distinct_initial_parents=int(result.parent_candidate_id.nunique()),
        macro_bin_count=len({k[2:] for k in counts}),
        complete_pairs_only=complete_pairs_only,population_weights=False)


def broad_firm_bank(store, *, split, max_rows, seed, calculated_fraction=.5):
    """Two-pass bounded index sampling; all time strata are eligible.

    Equal lineage -> equal nonempty time/macro bin -> equal eligible firm.
    The sampler never retains all per-firm states in RAM or GPU.
    """
    if not 0 <= calculated_fraction <= 1 or max_rows<2:
        raise ValueError('invalid explicit Firm sampling budget')
    # Pass one stores only counts per shard/lineage/stratum, not firm states.
    groups=defaultdict(list)
    for shard_index,capture in enumerate(store.captures()):
        if capture['split']!=split: raise ValueError('Firm split/lineage mismatch')
        valid=capture['eligible'].bool(); paths=capture['path']; context=capture['context']
        for pid in torch.unique(paths[valid]).tolist():
            ids=torch.nonzero(valid & (paths==pid)).flatten()
            if not len(ids): continue
            h,k=context[ids[0]].tolist()
            key=(int(pid),time_bin(capture['source_time']),*macro_bin(h,k))
            groups[key].append((shard_index,int(pid),len(ids)))
    if not groups: raise ValueError('NO_LEGAL_BROAD_FIRM_STATES')
    lineages=defaultdict(list)
    for key in sorted(groups): lineages[key[0]].append(key)
    rng=np.random.default_rng(seed); wanted=defaultdict(list)
    bands=sorted({key[1] for key in groups})
    by_band={band:sorted({key[0] for key in groups if key[1]==band}) for band in bands}
    ncal=round(max_rows*calculated_fraction)
    kinds=['CALCULATED_MACRO_COLLOCATION']*ncal+['ON_POLICY']*(max_rows-ncal)
    rng.shuffle(kinds)
    for output_index,kind in enumerate(kinds):
        band=bands[output_index%len(bands)];pids=by_band[band]
        pid=pids[int(rng.integers(len(pids)))];keys=[k for k in lineages[pid] if k[1]==band]
        key=keys[int(rng.integers(len(keys)))]; choices=groups[key]
        shard,pid,count=choices[int(rng.integers(len(choices)))]; offset=int(rng.integers(count))
        wanted[shard].append((output_index,pid,offset,kind))
    states=torch.empty(max_rows,7);contexts=torch.empty(max_rows,2)
    paths=torch.empty(max_rows,dtype=torch.long);times=torch.empty_like(paths)
    firms=torch.empty_like(paths); provenance=[None]*max_rows
    branches=torch.empty_like(paths)
    for shard_index,capture in enumerate(store.captures()):
        for out,pid,offset,kind in wanted.get(shard_index,[]):
            ids=torch.nonzero(capture['eligible'].bool() & (capture['path']==pid)).flatten()
            row=int(ids[offset])
            state=capture['states'][row].clone(); context=capture['context'][row].clone()
            if kind=='CALCULATED_MACRO_COLLOCATION': state[5:7]=context
            states[out]=state; contexts[out]=context; paths[out]=capture['path'][row]
            firms[out]=capture['firm_id'][row]; times[out]=capture['source_time'];provenance[out]=kind
            branches[out]=capture.get('source_branch',-1)
    return InnerFPCollocationBank(states,torch.zeros(max_rows,dtype=torch.long),seed,split,
        dict(path=paths,firm_id=firms,source_time=times,source_branch=branches,true_macro_context=contexts,
            source_kind=provenance,operator=VERSION,
            target_contract='REBUILD_CONDITIONAL_CHILD_BANK_AND_BELLMAN_RHS; NO_CACHED_LABELS',
            calculated_fraction=calculated_fraction,
            requested_rows=max_rows,actual_draws=max_rows,
            unique_source_firm_time_count=len(set(zip(paths.tolist(),times.tolist(),branches.tolist(),firms.tolist()))),
            replacement_draws_are_not_new_independent_states=True,
            available_firm_capture_rows=sum(n for values in groups.values() for _,_,n in values),
            available_time_bands=sorted({key[1] for key in groups}),
            sampler='EQUAL_NONEMPTY_TIME_BAND_THEN_LINEAGE_THEN_MACRO_STRATUM_THEN_CAPTURE_THEN_FIRM'))
