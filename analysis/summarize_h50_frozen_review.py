"""CPU-only aggregation of saved diagnostic evidence; no model optimization."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.nested_equilibrium_utils import atomic_write_json
from data.finite_direct_long_horizon_runtime import ShardStore
from data.finite_direct_long_horizon_sampling import sample_macro_rows,broad_firm_bank,time_bin
from data.finite_direct_macro_contract import select_training_rows


class CombinedStore:
    def __init__(self,stores):self.stores=stores
    def rows(self):
        for s in self.stores:yield from s.rows()
    def captures(self):
        for s in self.stores:yield from s.captures()


def summarize(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig=output/'figures';fig.mkdir(exist_ok=True)
    selected=pd.read_csv(root/'selected_parents.csv');events=pd.read_csv(root/'economic_events.csv')
    children=pd.read_csv(root/'all_child_rows.csv');n=len(selected)
    details=events[events.phase.isin(['INITIAL_PROCESSED','POST_DECISION'])].copy()
    coverage=[]
    for t,g in children.groupby('t'):
        valid=g[g.economic_state_valid.fillna(False)]
        pairs=sum(set(h.child_branch)=={0,1} for _,h in valid.groupby(['split','parent_candidate_id']))
        coverage.append(dict(t=t,scheduled_initial_parents=n,scheduled_child_rows=2*n,
            valid_child_rows=len(valid),valid_fraction=len(valid)/(2*n),complete_sdf_pairs=pairs,
            valid_initial_lineages=valid.parent_candidate_id.nunique(),
            attempted_child_rows=int(g.node_attempted.fillna(False).sum()),
            time_stratum=['EARLY','MIDDLE','LATE'][time_bin(t-1)]))
    coverage=pd.DataFrame(coverage);coverage.to_csv(output/'coverage_by_time.csv',index=False)
    # Detailed main spine plus explicitly missing suffix; NaN never means zero.
    timeline=[]
    for _,s in selected.iterrows():
        for t in range(51):
            candidates=details[(details.initial_parent_id==s.parent_id)&(details.t==t)&(details.branch==(-1 if t==0 else 0))]
            if len(candidates):r=candidates.iloc[0].to_dict()
            else:r=dict(initial_parent_id=s.parent_id,t=t,branch=0,phase='NOT_EXECUTED',entry_attempted=False,
                economic_state_valid=False,continuation_allowed=False,failure_reason='ANCESTOR_UNDEFINED_OR_INITIAL_ROOT_FAILURE')
            r.update(split=s.split,sampling_source=s.sampling_source)
            timeline.append(r)
    timeline=pd.DataFrame(timeline)
    timeline.to_csv(output/'macro_path_timeseries.csv',index=False)
    events.to_csv(output/'economic_timing_events.csv',index=False)
    # Diagnostics thresholds are descriptive, NOT economic acceptance gates.
    thresholds=dict(macro_error_absolute=np.log(2),hard_default_fraction=.5,capital_decline_fraction=.25,
        interpretation='EXPLORATORY_TIMELINE_THRESHOLDS_NOT_ROOT_OR_TRAINING_GATES')
    failures=[]
    for pid,g in timeline.groupby('initial_parent_id'):
        g=g.sort_values('t').copy();g['err_h']=g.hatcf_pred-g.hatc_cal;g['err_k']=g.lnkf_pred-g.lnk_cal
        g['large_default']=g.hard_default_count/g.alive_pre_decision.replace(0,np.nan)>=.5
        initialK=g[g.t==0].K_pre_exit.iloc[0]
        first=lambda mask:int(g.loc[mask,'t'].min()) if bool(mask.any()) else None
        failures.append(dict(initial_parent_id=pid,
            first_macro_prediction_error_t=first((g.err_h.abs()>=np.log(2))|(g.err_k.abs()>=np.log(2))),
            first_large_default_t=first(g.large_default),
            first_capital_decline_t=first(g.K_pre_exit<initialK*.75),
            first_entry_rejection_t=first(g.entry_attempted.fillna(False)&(g.potential_entry_count>0)&(g.accepted_entry_count==0)),
            first_no_entry_candidates_t=first(g.entry_attempted.fillna(False)&(g.potential_entry_count==0)),
            first_entry_not_executed_t=first((g.t>0)&~g.entry_attempted.fillna(False)),
            first_invalid_child_t=first((g.t>0)&~g.economic_state_valid.fillna(False)),
            first_no_legal_next_parent_t=first(~g.continuation_allowed.fillna(False)),
            last_valid_spine_time=int(g.loc[g.economic_state_valid.fillna(False),'t'].max()) if g.economic_state_valid.fillna(False).any() else None))
    pd.DataFrame(failures).to_csv(output/'first_failure_timeline.csv',index=False)
    entry=details[details.t>0].copy()
    entry['entry_status']=np.select([~entry.entry_attempted.fillna(False),entry.potential_entry_count==0,
        entry.accepted_entry_count>0],['ENTRY_NOT_EXECUTED','NO_CANDIDATES','POSITIVE_ENTRY'],default='ALL_REJECTED')
    entry.to_csv(output/'entry_timeseries.csv',index=False)
    quotas={};supports=[];firms=[]
    for split in ('train','validation'):
        stores=[ShardStore(p) for p in sorted((root/'raw').glob(split+'_*')) if (p/'manifest.json').exists()]
        combined=CombinedStore(stores)
        for sdf in (False,True):
            frame,audit=sample_macro_rows(combined,pair_budget=8192 if split=='train' else 2048,
                seed=20261021,complete_pairs_only=sdf)
            quotas[f'{split}_{"sdf" if sdf else "fc1"}']=audit
            frame.to_csv(output/f'{split}_{"sdf" if sdf else "fc1"}_sample.csv',index=False)
            if not sdf:
                for _,r in frame.iterrows():supports.append(dict(split=split,parent_id=r.parent_candidate_id,
                    t=int(r.t)-1,time_stratum=time_bin(int(r.t)-1),h=r.Hatc_cal_t,lnk=r.LnK_cal_t,
                    child_branch=r.child_branch,target_h=r.Hatc_cal_t1,target_lnk=r.LnK_cal_t1))
        bank=broad_firm_bank(combined,split=split,max_rows=512,seed=20261021)
        quotas[f'{split}_firm']=dict(bank.metadata,source_kind=None,path=None,firm_id=None,
            source_time=None,source_branch=None,true_macro_context=None)
        for j,state in enumerate(bank.parent_states):
            firms.append(dict(split=split,parent_id=int(bank.metadata['path'][j]),t=int(bank.metadata['source_time'][j]),
                firm_id=int(bank.metadata['firm_id'][j]),kind=bank.metadata['source_kind'][j],
                h=float(state[5]),lnk=float(state[6]),time_stratum=time_bin(int(bank.metadata['source_time'][j]))))
    pd.DataFrame(supports).to_csv(output/'fc1_macro_support.csv',index=False)
    pd.DataFrame(firms).to_csv(output/'firm_macro_support.csv',index=False)
    atomic_write_json(output/'sampling_quota_audit.json',quotas)
    # Plot population denominators, not success-conditioned survival alone.
    def line(name,series,ylabel):
        plt.figure(figsize=(8,4))
        for label,x,y in series:plt.plot(x,y,label=label,marker='.')
        plt.xlabel('Economic time');plt.ylabel(ylabel);plt.legend();plt.tight_layout();plt.savefig(fig/name,dpi=160);plt.close()
    line('valid_child_fraction_by_time.png',[('valid/(2N0)',coverage.t,coverage.valid_fraction)],'Valid child fraction')
    agg=timeline.groupby('t').alive_post_exit.agg(['mean','count'])
    line('surviving_firms_by_time.png',[('mean among observed spine nodes; missing != zero',agg.index,agg['mean'])],'Hard survivors')
    for key,name in [('lnk','forecast_vs_actual_lnk.png'),('hatc','forecast_vs_actual_hatc.png')]:
        pred='lnkf_pred' if key=='lnk' else 'hatcf_pred';actual=key+'_cal'
        plt.figure(figsize=(5,5));v=details[[pred,actual]].dropna();plt.scatter(v[pred],v[actual],s=8,c=details.loc[v.index,'t'],cmap='viridis')
        lo=float(v.min().min());hi=float(v.max().max());plt.plot([lo,hi],[lo,hi],'k--');plt.xlabel(pred);plt.ylabel(actual)
        plt.colorbar(label='t');plt.tight_layout();plt.savefig(fig/name,dpi=160);plt.close()
    a=entry.groupby('t')[['potential_entry_count','accepted_entry_count']].sum()
    line('entry_candidates_vs_accepted.png',[(k,a.index,a[k]) for k in a],'Observed child entry counts')
    plt.figure(figsize=(8,4));sp=pd.DataFrame(supports)
    for band,label in enumerate(['EARLY','MIDDLE','LATE']):
        v=sp[sp.time_stratum==band];plt.scatter(v.h,v.lnk,s=10,label=f'{label}: {len(v)} child rows')
    plt.xlabel('Cleared/calculated parent h');plt.ylabel('Calculated parent lnK');plt.legend();plt.tight_layout()
    plt.savefig(fig/'h_lnk_coverage_early_middle_late.png',dpi=160);plt.close()
    roots=pd.read_csv(root/'initial_hstar_feasibility.csv');roots.to_csv(output/'initial_hstar_feasibility.csv',index=False)
    roots[~roots.root_found.fillna(False)].to_csv(output/'root_failure_summary.csv',index=False)
    root_success=int(roots.root_found.sum());late=int(coverage[coverage.t>=11].valid_child_rows.sum())
    initial=details[details.t==0];attempted=children[children.node_attempted.fillna(False)]
    summary=dict(real_parent_count=n,root_found_count=root_success,root_failed_count=n-root_success,
        horizon=50,valid_child_count=int(coverage.valid_child_rows.sum()),complete_sdf_pair_count=int(coverage.complete_sdf_pairs.sum()),
        max_valid_economic_time=int(coverage.loc[coverage.valid_child_rows>0,'t'].max()),
        middle_late_coverage='ACHIEVED' if late else 'NOT_ACHIEVED',
        initial_hard_default_mean=float(initial.hard_default_count.mean()),
        initial_hard_default_fraction=float(initial.hard_default_count.sum()/initial.alive_pre_decision.sum()),
        initial_default_range=[int(initial.hard_default_count.min()),int(initial.hard_default_count.max())],
        attempted_failure_counts=attempted.failure_reason.fillna('VALID').value_counts().to_dict(),
        entry_status_counts=entry.entry_status.value_counts().to_dict(),
        timeline_thresholds=thresholds,training_performed=False,optimizer_steps=0,A800_COST='NOT_MEASURED',
        ready_for_formal_training=False)
    atomic_write_json(output/'diagnostic_summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(summarize(a.input,a.output))
