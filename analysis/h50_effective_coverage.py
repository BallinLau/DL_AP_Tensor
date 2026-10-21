"""Scalar streaming lineage coverage. No optimizer, NN call or fake labels."""
from collections import Counter,defaultdict
import math
from pathlib import Path
import numpy as np
import pandas as pd
from data.finite_direct_macro_contract import select_training_rows
from data.finite_direct_joint_coverage import joint_bin,bin_direction


def lineage_statistics(weights,sources=()):
    w=np.array(list(weights),float);total=w.sum()
    shares=np.sort(w/total)[::-1] if total else np.array([])
    sc=np.array(list(Counter(sources).values()),float);sc=sc/sc.sum() if sc.sum() else sc
    return dict(raw_rows=int(total),unique_lineages=len(w),ESS=float(total*total/(w@w)) if total else 0.,
        largest_lineage_share=float(shares[0]) if len(shares) else None,top5_lineage_share=float(shares[:5].sum()) if len(shares) else None,
        source_entropy=float(-(sc*np.log(sc)).sum()) if len(sc) else None)


def collect_effective_coverage(store,output,cfg):
    groups=defaultdict(Counter);pair_counts=Counter();cells=defaultdict(set);sources=defaultdict(Counter);last={};directions=Counter()
    for frame in store.rows():
        selected=select_training_rows(frame)
        for r in selected.fc1.to_dict('records'):
            pid=int(r['parent_candidate_id']);t=int(r['t']);pt=t-1
            band='EARLY_0_9' if pt<10 else 'MIDDLE_10_29' if pt<30 else 'LATE_30_49'
            cell=joint_bin(r['Hatc_cal_t'],r['LnK_cal_t'],cfg)
            if cell is None:raise ValueError('invalid macro label admitted into coverage')
            direction=bin_direction(cell,cfg);directions[direction]+=1
            source=r.get('sampling_source','UNKNOWN');family=r.get('b_distribution_family','UNKNOWN')
            keys=[('ALL','ALL'),('TIME',str(t)),('MACRO_BIN',str(cell)),('FAMILY',family),('TIME_STRATUM',band),('SOURCE',source)]
            for key in keys:
                groups[key][pid]+=1;cells[key].add(cell);sources[key][source]+=1
            last[pid]=max(last.get(pid,0),t)
        for _,g in selected.sdf_pairs.groupby(['split','parent_candidate_id','t']):
            r=g.iloc[0];pt=int(r.t)-1;band='EARLY_0_9' if pt<10 else 'MIDDLE_10_29' if pt<30 else 'LATE_30_49'
            for key in [('ALL','ALL'),('TIME',str(int(r.t))),('MACRO_BIN',str(joint_bin(r.Hatc_cal_t,r.LnK_cal_t,cfg))),
                        ('FAMILY',r.get('b_distribution_family','UNKNOWN')),('TIME_STRATUM',band),('SOURCE',r.get('sampling_source','UNKNOWN'))]:pair_counts[key]+=1
    for band in ('EARLY_0_9','MIDDLE_10_29','LATE_30_49'):groups['TIME_STRATUM',band]
    rows=[]
    for key,weights in groups.items():
        stats=lineage_statistics(weights.values())
        sc=np.array(list(sources[key].values()),float);sc=sc/sc.sum() if sc.sum() else sc
        stats['source_entropy']=float(-(sc*np.log(sc)).sum()) if len(sc) else None
        end=[last[p] for p in weights]
        rows.append(dict(group=key[0],stratum=key[1],**stats,complete_sdf_pairs=pair_counts[key],
            joint_macro_occupied_bins=len(cells[key]),last_valid_t_mean=float(np.mean(end)) if end else None,
            last_valid_t_median=float(np.median(end)) if end else None))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame(rows);result.to_csv(output/'effective_macro_coverage.csv',index=False)
    for group,name in [('TIME','unique_lineages_by_time'),('MACRO_BIN','unique_lineages_by_macro_bin'),('FAMILY','unique_lineages_by_family'),('TIME_STRATUM','time_stratum_coverage')]:
        result[result.group==group].to_csv(output/(name+'.csv'),index=False)
    pd.DataFrame([dict(overflow_direction=k,legal_fc1_rows=v) for k,v in directions.items()]).to_csv(output/'directional_overflow.csv',index=False)
    return result
