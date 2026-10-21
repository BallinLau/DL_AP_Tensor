"""Streaming scalar diagnostics of the ORIGINAL H50 operator; no NN calls."""
from pathlib import Path
from collections import Counter,defaultdict
import json,math,time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.nested_equilibrium_utils import atomic_write_json,stable_json_hash
from data.finite_direct_joint_coverage import JointCoverageConfig,joint_bin


def _defined(r):
    return all(math.isfinite(r.get(k,float('nan'))) for k in ('Hatc_cal_t1','LnK_cal_t1','C_raw_t1','K_cal_t1')) and r.get('C_raw_t1',0)>0 and r.get('K_cal_t1',0)>0


def error_metrics(values):
    a=np.asarray(values,dtype=float)
    a=a[np.isfinite(a)]
    return dict(metric_valid_count=len(a),MAE=float(np.abs(a).mean()) if len(a) else np.nan,
        RMSE=float(np.sqrt((a*a).mean())) if len(a) else np.nan,
        signed_bias=float(a.mean()) if len(a) else np.nan,
        **{f'P{q}_absolute_error':float(np.quantile(np.abs(a),q/100)) if len(a) else np.nan for q in (50,90,99)},
        data_status='AVAILABLE' if len(a) else 'NO_VALID_DATA')


def write_dashboard(store,output,cfg=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);cfg=cfg or JointCoverageConfig()
    identity=stable_json_hash(dict(provenance=store.manifest['provenance'],shards=store.manifest['shards'],config=vars(cfg)))
    done=output/'dashboard_manifest.json'
    if done.exists() and json.loads(done.read_text())['identity']==identity:return pd.read_csv(output/'path_survival_by_time.csv')
    started=time.monotonic();N=store.manifest['scheduled_initial_parents'];prov=store.manifest['provenance']
    outer=prov['outer'];stage=prov['stage'];split=prov['split'];counters=defaultdict(Counter)
    quantiles=defaultdict(lambda:defaultdict(list));errors=defaultdict(list);failures=Counter();bins=Counter()
    source={};initial={};main={};terminal={};previous={};links=[];family_counts=Counter();coverage_sources=Counter()
    for shard in store.chunks():
        rows=shard['rows'];time_=rows[0]['t'] if rows else 0;arrival=counters[time_]
        arrival['attempted_parent_paths']+=len(shard.get('attempted_parent_ids',[]))
        attempted=set(shard.get('attempted_parent_ids',[]))
        for r in rows:
            pid=r['parent_candidate_id'];source[pid]=(r.get('sampling_source','UNKNOWN'),r.get('b_distribution_family','UNKNOWN'))
        for r in shard.get('parent_records',[]):
            pid=r['parent_candidate_id']
            if pid in attempted and _defined(r):arrival['macro_defined_parent_paths']+=1
            if r['t']==0:
                initial[pid]=r;main[pid,0]=r
                family_counts[source[pid][1]]+=1
                b=joint_bin(r.get('Hatc_cal_t1',np.nan),r.get('LnK_cal_t1',np.nan),cfg)
                if b is not None:bins['HSTAR_VALID',source[pid][0],source[pid][1],*b]+=1
                if r.get('economic_state_valid') and b is not None:bins['ECONOMIC_VALID',source[pid][0],source[pid][1],*b]+=1
        per_parent=defaultdict(list)
        for r in rows:
            pid=r['parent_candidate_id'];t=r['t'];branch=r['child_branch'];label,family=source[pid]
            if r.get('economic_state_valid'):per_parent[pid].append(branch)
            if not r.get('economic_state_valid'):failures[t,r.get('node_failure_reason',r.get('failure_reason','UNKNOWN'))]+=1
            if branch==0:
                main[pid,t]=r
                if not r.get('continuation_allowed',False) and pid not in terminal:terminal[pid]=t
                prior=previous.get(pid)
                if prior is not None and t==prior['t']+1:
                    links.append(dict(parent_id=pid,t=prior['t'],next_t=t,
                        err_h=prior.get('Hatcf_t1',np.nan)-prior.get('Hatc_cal_t1',np.nan),
                        err_K=prior.get('LnKF_t1',np.nan)-prior.get('LnK_cal_t1',np.nan),
                        next_hard_default_count=r.get('hard_default_count'),next_alive_predecision=r.get('predecision_alive_firms'),
                        next_node_attempted=r.get('node_attempted',False),causal_interpretation='ASSOCIATION_ONLY'))
                previous[pid]=r
            for metric in ('hard_default_count','entry_candidate_count','entry_accepted_count'):
                if r.get(metric) is not None:arrival['both_children_'+metric]+=r[metric]
            if r.get('economic_state_valid'):
                arrival[f'valid_child{branch}_rows']+=1
                b=joint_bin(r['Hatc_cal_t'],r['LnK_cal_t'],cfg)
                bins['FC1_USABLE_CHILD_WEIGHTED',label,family,*b]+=1
                coverage_sources['FC1_ROWS',label]+=1
            b=joint_bin(r.get('Hatc_cal_t',np.nan),r.get('LnK_cal_t',np.nan),cfg)
            for kind,legal in [('TRUE_STATE_ONE_STEP',r.get('economic_state_valid',False)),('ON_POLICY_BELIEF_GAP',_defined(r))]:
                if not legal:continue
                for variable,pred,target in [('h','Hatcf_t1','Hatc_cal_t1'),('ell','LnKF_t1','LnK_cal_t1')]:
                    value=r.get(pred,np.nan)-r.get(target,np.nan)
                    errors[kind,t,'ALL','ALL',variable].append(value)
                    errors[kind,t,str(b),family,variable].append(value)
        for pid,branches in per_parent.items():
            arrival['paths_with_at_least_one_valid_child']+=1
            if set(branches)=={0,1}:
                arrival['paths_with_two_valid_children']+=1;arrival['complete_sdf_pairs']+=1
                coverage_sources['SDF_PAIRS',source[pid][0]]+=1
    tables=[]
    for t in range(51):
        c=counters[t];rs=[r for (pid,time_),r in main.items() if time_==t]
        active=sum(r.get('continuation_allowed',_defined(r)) for r in rs)
        r=dict(outer=outer,stage=stage,split=split,t=t,parent_time=max(0,t-1),scheduled_initial_paths=N,
            hstar_root_pass_paths=len(initial),attempted_parent_paths=c['attempted_parent_paths'] if t else len(initial),
            macro_defined_parent_paths=c['macro_defined_parent_paths'] if t else sum(map(_defined,initial.values())),
            main_spine_alive_paths=active,main_spine_macro_defined_paths=sum(_defined(r) for r in rs if r.get('continuation_allowed',_defined(r))),
            main_spine_alive_fraction=active/N if N else np.nan,
            paths_terminated_this_time=sum(time_==t for time_ in terminal.values()),
            cumulative_terminated_paths=sum(time_<=t for time_ in terminal.values()),
            right_censored_paths=active if t==50 else 0,
            timing='t0=ROOT_NODE; t>=1=CHILD_ARRIVAL; incoming-parent time=t-1',
            failure_reason_counts=json.dumps({reason:n for (time_,reason),n in failures.items() if time_==t},sort_keys=True))
        for name in ('paths_with_at_least_one_valid_child','paths_with_two_valid_children','valid_child0_rows','valid_child1_rows','complete_sdf_pairs'):
            r[name]=c[name]
        for field in ('predecision_alive_firms','hard_default_count','post_exit_alive_firms','entry_candidate_count','entry_accepted_count'):
            r[field]=sum(q[field] for q in rs if q.get(field) is not None)
        for label,condition in [('entry_positive_nodes',lambda q:q.get('entry_accepted_count',0)>0),
            ('entry_no_candidates_nodes',lambda q:q.get('entry_candidate_count',-1)==0),
            ('entry_all_rejected_nodes',lambda q:q.get('entry_candidate_count',0)>0 and q.get('entry_accepted_count',-1)==0)]:
            r[label]=sum(condition(q) for q in rs if q.get('entry_attempted',False))
        r['entry_not_executed_paths']=sum(not q.get('entry_attempted',False) for q in rs)
        for field,key in [('K_cal','K_cal_t1'),('C_raw','C_raw_t1'),('h_cal','Hatc_cal_t1'),('ell_cal','LnK_cal_t1'),('h_f','Hatcf_t1'),('ell_f','LnKF_t1')]:
            values=[q[key] for q in rs if math.isfinite(q.get(key,np.nan))]
            for p in (10,50,90):r[f'{field}_p{p}']=float(np.quantile(values,p/100)) if values else np.nan
        r.update({k:c[k] for k in c if k.startswith('both_children_')});tables.append(r)
    survival=pd.DataFrame(tables);survival.to_csv(output/'path_survival_by_time.csv',index=False)
    survival.to_csv(output/'entry_exit_by_time.csv',index=False)
    error_rows=[]
    for kind in ('TRUE_STATE_ONE_STEP','ON_POLICY_BELIEF_GAP'):
        for t in range(1,51):
            for variable in ('h','ell'):
                key=(kind,t,'ALL','ALL',variable);errors.setdefault(key,[])
    for (kind,t,bin_,family,variable),values in errors.items():
        metrics=error_metrics(values);metrics.update(outer=outer,stage=stage,split=split,t=t,
            diagnostic_kind=kind,macro_coverage_bin=bin_,b_distribution_family=family,variable=variable,
            original_initial_path_denominator=N,scheduled_child_denominator=2*N,
            dropped_or_terminated_count=2*N-metrics['metric_valid_count'])
        error_rows.append(metrics)
    error_frame=pd.DataFrame(error_rows);error_frame.to_csv(output/'macro_forecast_error_by_time.csv',index=False)
    pd.DataFrame(links).to_csv(output/'forecast_error_next_default.csv',index=False)
    pd.DataFrame([dict(phase=phase,sampling_source=source_,family=family,h_bin=h,ell_bin=k,count=n)
        for (phase,source_,family,h,k),n in bins.items()]).to_csv(output/'coverage_2d_bins.csv',index=False)
    composition=[dict(layer='INITIAL_PARENTS',sampling_source=label,count=sum(v[0]==label for v in source.values())) for label in ('BETA','EMPIRICAL')]
    composition.extend(dict(layer=layer,sampling_source=label,count=n) for (layer,label),n in coverage_sources.items())
    pd.DataFrame(composition).to_csv(output/'coverage_source_composition.csv',index=False)
    pd.DataFrame([dict(t=t,reason=reason,count=n) for (t,reason),n in failures.items()]).to_csv(output/'failure_reasons.csv',index=False)
    last=[]
    for pid in source:
        times=[t for (p,t),r in main.items() if p==pid and r.get('economic_state_valid',False)]
        last.append(dict(parent_id=pid,last_valid_t=max(times) if times else np.nan,
            terminated_t=terminal.get(pid),right_censored=bool(main.get((pid,50),{}).get('continuation_allowed',False))))
    pd.DataFrame(last).to_csv(output/'path_endpoints.csv',index=False)
    metadata=dict(outer=outer,stage_name=stage,split=split,scheduled_initial_paths=N,
        firm_checkpoint_hash=prov['model_hashes']['firm'],fc1_checkpoint_hash={k:prov['model_hashes'][k] for k in ('fc1_c','fc1_k')},
        sdf_checkpoint_hash=prov['model_hashes']['sdf'],wealth_checkpoint_hash=prov['model_hashes']['wealth'],
        initial_bank_sha=prov.get('diagnostic_metadata',{}).get('initial_bank_sha',prov['parent_bank_hash']),
        initial_parent_ids_hash=stable_json_hash(sorted(source)),rng_or_crn_identifier=prov['crn_hash'],
        economic_operator=prov['operator'],horizon=50,validity_contract='FINITE_DIRECT_EXISTING_NO_RELAXATION',
        economic_config_hash=prov.get('economic_hash'),root_config=prov.get('root_config'),
        runtime_source_sha256=prov.get('runtime_source_sha256'),
        bins=vars(cfg),recursive_forecast_state='NOT_TESTED_DISABLED_IN_EXISTING_SETUP',
        true_state_vs_on_policy='EXPAND_USES_TRUE_PARENT_MACRO; A_FILTERS_TRAINABLE_ROWS; B_INCLUDES_DEFINED_LEDGER_NODES',
        collection_seconds=time.monotonic()-started,training_performed=False)
    atomic_write_json(output/'stage_provenance.json',metadata)
    draw_figures(output,survival,error_frame,bins,composition,family_counts,cfg)
    atomic_write_json(done,dict(identity=identity,metadata=metadata))
    return survival


def draw_figures(output,survival,errors,bins,composition,families,cfg):
    def save(name):plt.tight_layout();plt.savefig(output/name,dpi=140);plt.close()
    he,ke=cfg.edges()
    def panel(ax,phase,source=None):
        hist=np.zeros((cfg.n_h_bins,cfg.n_ell_bins))
        for (p,s,f,h,k),n in bins.items():
            if p==phase and (source is None or s==source) and h>=0 and k>=0:hist[h,k]+=n
        ax.pcolormesh(he,ke,np.log1p(hist.T),shading='auto');ax.set(xlabel='h_cal',ylabel='ell_cal',xlim=cfg.h_bounds,ylim=cfg.ell_bounds)
        if not hist.sum():ax.text(.5,.5,'NO VALID DATA',transform=ax.transAxes,ha='center')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    axes[0].set(xlim=cfg.h_bounds,ylim=cfg.ell_bounds,xlabel='h_cal',ylabel='ell_cal',title='Proposed')
    axes[0].text(.5,.5,'h_cal undefined before strict root\nSee candidate_acceptance.csv',transform=axes[0].transAxes,ha='center')
    for ax,phase in zip(axes[1:],['HSTAR_VALID','FC1_USABLE_CHILD_WEIGHTED']):panel(ax,phase);ax.set_title(phase)
    save('joint_macro_coverage.png')
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,source in zip(axes,['BETA','EMPIRICAL']):panel(ax,'FC1_USABLE_CHILD_WEIGHTED',source);ax.set_title(source)
    save('beta_vs_empirical_coverage.png')
    plt.figure(figsize=(8,4));plt.bar(list(families),list(families.values()));plt.xticks(rotation=30);plt.ylabel('Initial Parents');save('b_distribution_families.png')
    plt.figure(figsize=(8,4));plt.plot(survival.t,survival.paths_with_at_least_one_valid_child,label='at least one')
    plt.plot(survival.t,survival.paths_with_two_valid_children,label='two valid');plt.plot(survival.t,survival.complete_sdf_pairs,'--',label='complete SDF');plt.legend();save('valid_children_and_pairs.png')
    for variable,name in [('h','fc1_hatc_error_by_time.png'),('ell','fc1_lnk_error_by_time.png')]:
        f=errors[(errors.diagnostic_kind=='TRUE_STATE_ONE_STEP')&(errors.macro_coverage_bin=='ALL')&(errors.variable==variable)]
        fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True)
        for col in ('MAE','signed_bias','P90_absolute_error'):axes[0].plot(f.t,f[col],label=col)
        axes[0].legend();axes[0].set_title('TRUE_STATE_ONE_STEP; NaN = no valid data')
        axes[1].plot(f.t,f.metric_valid_count,label='valid sample count');axes[1].legend();axes[1].set_xlabel('Child time');save(name)
    plt.figure(figsize=(8,4))
    for field in ('hard_default_count','entry_candidate_count','entry_accepted_count','post_exit_alive_firms'):plt.plot(survival.t,survival[field],label=field)
    plt.title('Main spine nodes, not both siblings');plt.legend();save('default_entry_by_time.png')
    fig,ax=plt.subplots(figsize=(8,4));initial=[r for r in composition if r['layer']=='INITIAL_PARENTS'];counts={r['sampling_source']:r['count'] for r in initial}
    actual=counts.get('EMPIRICAL',0)/max(1,sum(counts.values()));ax.bar(['target empirical','actual initial empirical','beta shortfall fill'],[.5,actual,max(0,.5-actual)])
    ax.set_ylim(0,1);ax.set_title('Current stage only; not future Outer evidence');save('empirical_share_by_outer.png')


def compare_validation_stages(stage_directories,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);loaded={}
    for name,path in stage_directories.items():
        path=Path(path)
        if (path/'stage_provenance.json').exists():loaded[name]=(json.loads((path/'stage_provenance.json').read_text()),pd.read_csv(path/'path_survival_by_time.csv'))
    fields=('initial_bank_sha','initial_parent_ids_hash','rng_or_crn_identifier','economic_operator','horizon','validity_contract','scheduled_initial_paths',
        'economic_config_hash','root_config','runtime_source_sha256','bins')
    reference=next(iter(loaded.values()))[0] if loaded else None
    for p,_ in loaded.values():
        if p['split']!='validation' or any(p[k]!=reference[k] for k in fields):raise ValueError('UNFAIR_D0_D1_D2_VALIDATION_COMPARISON')
    records=[]
    for name in ('D0','D1','D2'):
        if name not in loaded:records.append(dict(stage=name,status='NO_ACTUAL_STAGE_DATA'));continue
        p,s=loaded[name];row=dict(stage=name,status='ACTUAL_FROZEN_STAGE',total_initial_paths=p['scheduled_initial_paths'])
        for t in (1,5,10,20,30,50):row[f'survived_t{t}']=int(s.loc[s.t==t,'main_spine_alive_paths'].iloc[0])
        endpoint_path=Path(stage_directories[name])/'path_endpoints.csv'
        ends=pd.read_csv(endpoint_path) if endpoint_path.exists() else pd.DataFrame()
        values=ends.last_valid_t.dropna().to_numpy() if 'last_valid_t' in ends else np.array([])
        for q in (10,50,90):row[f'last_valid_t_p{q}']=float(np.percentile(values,q)) if len(values) else np.nan
        row['right_censored_paths']=int(ends.right_censored.sum()) if 'right_censored' in ends else None
        row['attribution']='SYSTEM_CHANGE_NOT_FC1_ONLY' if p['firm_checkpoint_hash']!=reference['firm_checkpoint_hash'] or p['sdf_checkpoint_hash']!=reference['sdf_checkpoint_hash'] else 'MATCHED_FIRM_SDF; CHECK_FC1_HASH'
        records.append(row)
    pd.DataFrame(records).to_csv(output/'comparison_d0_d1_d2.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,(p,s) in loaded.items():axes[0].plot(s.t,s.main_spine_alive_paths,label=name);axes[1].plot(s.t,s.main_spine_alive_fraction,label=name)
    for ax in axes:
        if loaded:ax.legend()
        else:ax.text(.5,.5,'NO ACTUAL D0/D1/D2 DATA\nNo training performed',transform=ax.transAxes,ha='center')
        ax.set_xlabel('t=0..50')
    axes[0].set_ylabel('continuable main spine count');axes[1].set_ylabel('/ original initial paths')
    fig.tight_layout();fig.savefig(output/'path_survival_d0_d1_d2.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,name in zip(axes,('D0','D1','D2')):
        ax.set_title(name)
        table=Path(stage_directories[name])/'coverage_2d_bins.csv' if name in loaded else None
        if table is None or not table.exists():
            ax.text(.5,.5,'NO ACTUAL STAGE DATA',ha='center',transform=ax.transAxes);continue
        table=pd.read_csv(table);p=loaded[name][0];cfg=p['bins']
        hist=np.zeros((cfg['n_h_bins'],cfg['n_ell_bins']))
        if len(table):
            for r in table[table.phase=='FC1_USABLE_CHILD_WEIGHTED'].itertuples():
                if r.h_bin>=0 and r.ell_bin>=0:hist[int(r.h_bin),int(r.ell_bin)]+=r.count
        ax.pcolormesh(np.linspace(*cfg['h_bounds'],cfg['n_h_bins']+1),np.linspace(*cfg['ell_bounds'],cfg['n_ell_bins']+1),np.log1p(hist.T))
        ax.set(xlabel='h_cal',ylabel='ell_cal')
    fig.tight_layout();fig.savefig(output/'macro_coverage_before_after_fc1.png',dpi=140);plt.close(fig)
    atomic_write_json(output/'comparison_provenance.json',dict(stages={k:v[0] for k,v in loaded.items()},fixed_conditions=list(fields),training_performed=False))
