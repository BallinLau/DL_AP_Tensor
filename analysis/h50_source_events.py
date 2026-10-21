"""Source/time node counters from immutable ledger shards only."""
from collections import Counter,defaultdict
from pathlib import Path
import pandas as pd


def source_event_table(store,output):
    counts=defaultdict(Counter);members=defaultdict(set);terminated={};pair_counts=Counter()
    prov=store.manifest['provenance']
    for shard in store.chunks():
        rows=shard['rows'];labels={r['parent_candidate_id']:r.get('source_type',r.get('sampling_source','UNKNOWN')) for r in rows}
        for pid,label in labels.items():members[label].add(pid)
        groups=defaultdict(list)
        for r in rows:groups[r['parent_candidate_id'],r['t']].append(r)
        for (pid,t),group in groups.items():
            label=labels[pid];c=counts[label,t]
            valid=sum(bool(r.get('economic_state_valid')) for r in group)
            c['at_least_one_valid_child']+=valid>0;c['valid_child_rows']+=valid
            c['complete_sdf_pairs']+=valid==2
            main=next(r for r in group if r['child_branch']==0)
            if not main.get('continuation_allowed',False) and pid not in terminated:terminated[pid]=t
        main_rows=[r for r in rows if r['child_branch']==0]
        main_rows+=shard.get('parent_records',[]) if rows and rows[0]['t']==1 else []
        for r in main_rows:
            pid=r['parent_candidate_id'];t=r.get('t',0);c=counts[labels[pid],t]
            c['main_spine_alive_paths']+=bool(r.get('continuation_allowed',False))
            c['macro_defined_paths']+=bool(r.get('macro_defined',False))
            c['entry_executed_nodes']+=bool(r.get('entry_attempted',False))
            for field in ('hard_default_count','predecision_alive_firms','post_exit_alive_firms','entry_candidate_count','entry_accepted_count'):
                if r.get(field) is not None:c[field]+=r[field]
    records=[]
    metadata=prov.get('diagnostic_metadata',{})
    for label,ids in members.items():
        for t in range(51):
            c=counts[label,t]
            records.append(dict(outer=prov['outer'],stage=prov['stage'],split=prov['split'],t=t,source_type=label,
                model_version=metadata.get('model_version','STAGE_MODEL_HASH_BOUND'),checkpoint_sha256=metadata.get('checkpoint_sha256','NOT_AVAILABLE_UNCOMMITTED_STAGE_MODEL'),
                original_initial_paths=len(ids),cumulative_terminated_paths=sum(terminated.get(pid,51)<=t for pid in ids),
                **{field:c[field] for field in ('main_spine_alive_paths','macro_defined_paths','at_least_one_valid_child','valid_child_rows','complete_sdf_pairs',
                    'hard_default_count','predecision_alive_firms','post_exit_alive_firms','entry_executed_nodes','entry_candidate_count','entry_accepted_count')}))
    frame=pd.DataFrame(records);Path(output).mkdir(parents=True,exist_ok=True);frame.to_csv(Path(output)/'node_metrics_by_source_time.csv',index=False)
    return frame
