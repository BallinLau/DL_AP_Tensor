"""CPU-only summaries of saved child ledgers and committed stage results.

No models, policies, economic operators or optimizers are called here. The
original observations (including failures) remain untouched in their shards.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis.nested_equilibrium_utils import atomic_write_bytes,stable_json_hash
from data.finite_direct_sampling import file_hash

POST_SDF='POST_SDF_VALIDATION_RESIMULATION'
METRICS=('G_mean','G_median','G_p90','G_p99','Hatc_forecast_MAE','LnK_forecast_MAE',
    'Delta_LnK_R2','FC1_C_R2','FC1_K_R2','Surviving_firm_count_mean','K_mean','C_mean',
    'Default_rate','Investment_rate','Entry_rate','Valid_child_fraction','Complete_SDF_pair_fraction')


def _values(frame,name):
    if name not in frame: return np.full(len(frame),np.nan)
    return pd.to_numeric(frame[name],errors='coerce').to_numpy(dtype=float)


def _mean(values):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def _r2(actual,predicted):
    actual=np.asarray(actual); predicted=np.asarray(predicted)
    finite=np.isfinite(actual)&np.isfinite(predicted)
    actual,predicted=actual[finite],predicted[finite]
    if len(actual)<2: return np.nan
    denominator=float(np.sum((actual-actual.mean())**2))
    return 1-float(np.sum((predicted-actual)**2))/denominator if denominator>0 else np.nan


def _rate(frame,numerator):
    x=_values(frame,numerator); weight=_values(frame,'firm_mass')
    finite=np.isfinite(x)&np.isfinite(weight)&(weight>0)
    return float(x[finite].sum()/weight[finite].sum()) if finite.any() else np.nan


def economic_metrics(frame,schedule,*,outer,stage):
    """Three source groups, with explicit scheduled/attempted/valid coverage.

    G_* uses saved normalized_G_at_prediction (same convention as the previous
    FC1 experiment). G_raw_mean/G_raw_abs_mean retain the signed physical ledger
    scale. Rates are pooled saved gate/entry masses over saved pre-decision
    firm_mass, not newly inferred hard-decision rates. Raw C/K/survival/mass
    statistics include failed rows WHERE the saved account is defined; valid-
    only counterparts are separate. Undefined later states are never imputed.
    """
    if frame.duplicated(['parent_candidate_id','child_branch','t']).any():
        raise ValueError('duplicate child in economic history')
    source={r['parent_id']:r['sampling_source'] for r in schedule['parents']}
    if not set(source.values()).issubset({'BETA','EMPIRICAL'}): raise ValueError('unknown sampling source')
    if len(source)!=len(schedule['parents']): raise ValueError('duplicate scheduled Parent')
    if not set(frame.parent_candidate_id).issubset(source): raise ValueError('unscheduled Parent in metric rows')
    if not frame.split.eq(schedule['split']).all(): raise ValueError('metric split mismatch')
    if not frame.economic_state_valid.isin([True,False]).all(): raise ValueError('invalid economic validity flag')
    labels=frame.parent_candidate_id.map(source)
    crn_hash=stable_json_hash([r['child_shock_keys'] for r in schedule['parents']])
    rows=[]
    for group in ('BETA','EMPIRICAL','ALL'):
        selected=frame if group=='ALL' else frame.loc[labels==group]
        expected=sum(group=='ALL' or kind==group for kind in source.values())
        valid=selected.loc[selected.economic_state_valid.astype(bool)]
        pair_counts=valid.groupby(['parent_candidate_id','t']).size()
        row=dict(outer=outer,stage=stage,split=schedule['split'],sampling_source=group,
            crn_hash=crn_hash,empirical_semantics='RECORDED_INITIAL_PARENT_DISTRIBUTION',
            scheduled_parent_count=expected,scheduled_child_count=2*expected,
            attempted_parent_count=int(selected.parent_candidate_id.nunique()),attempted_child_count=len(selected),
            valid_parent_count=int(valid.parent_candidate_id.nunique()),valid_child_count=len(valid),
            failed_child_count=int((~selected.economic_state_valid.astype(bool)).sum()),
            complete_sdf_pair_count=int((pair_counts==2).sum()),
            G_definition='normalized_G_at_prediction; valid finite economic nodes only',
            rate_definition='pooled saved mass / saved pre-decision firm_mass; gate-weighted, not hard rates')
        fraction=lambda n,d:float(n/d) if d else np.nan
        row.update(Parent_attempt_fraction=fraction(row['attempted_parent_count'],expected),
            Child_attempt_fraction=fraction(len(selected),2*expected),
            Valid_parent_fraction=fraction(row['valid_parent_count'],expected),
            Valid_child_fraction=fraction(len(valid),2*expected),
            Complete_SDF_pair_fraction=fraction(row['complete_sdf_pair_count'],expected))
        g=_values(valid,'normalized_G_at_prediction'); g=g[np.isfinite(g)]
        row.update(G_mean=_mean(g),G_finite_child_count=len(g),
            G_raw_mean=_mean(_values(valid,'G_at_prediction')),
            G_raw_abs_mean=_mean(np.abs(_values(valid,'G_at_prediction'))))
        for name,q in [('G_median',.5),('G_p90',.9),('G_p99',.99)]:
            row[name]=float(np.quantile(g,q)) if len(g) else np.nan
        h=_values(valid,'Hatc_cal_t1'); hf=_values(valid,'Hatcf_t1')
        k=_values(valid,'LnK_cal_t1'); kf=_values(valid,'LnKF_t1'); prev=_values(valid,'LnK_cal_t')
        row.update(Hatc_forecast_MAE=_mean(np.abs(hf-h)),LnK_forecast_MAE=_mean(np.abs(kf-k)),
            Hatc_forecast_finite_count=int((np.isfinite(h)&np.isfinite(hf)).sum()),
            LnK_forecast_finite_count=int((np.isfinite(k)&np.isfinite(kf)).sum()),
            Delta_LnK_R2=_r2(k-prev,kf-prev),FC1_C_R2=_r2(h,hf),FC1_K_R2=_r2(k,kf),
            Surviving_firm_count_mean=_mean(_values(selected,'surviving_mass')),
            K_mean=_mean(_values(selected,'K_cal_t1')),C_mean=_mean(_values(selected,'C_raw_t1')),
            Default_rate=_rate(selected,'default_mass'),Investment_rate=_rate(selected,'investment_mass'),
            Entry_rate=_rate(selected,'entry_mass'),
            Surviving_firm_count_mean_valid=_mean(_values(valid,'surviving_mass')),
            K_mean_valid=_mean(_values(valid,'K_cal_t1')),C_mean_valid=_mean(_values(valid,'C_raw_t1')),
            Default_rate_valid=_rate(valid,'default_mass'),Investment_rate_valid=_rate(valid,'investment_mass'),
            Entry_rate_valid=_rate(valid,'entry_mass'))
        for label,col in [('Surviving_firm_count','surviving_mass'),('K','K_cal_t1'),('C','C_raw_t1')]:
            row[label+'_finite_child_count']=int(np.isfinite(_values(selected,col)).sum())
        row['raw_economic_statistics_scope']='all attempted rows with defined saved accounts, including failures; _valid columns are conditional'
        row['metric_status']=json.dumps({k:'AVAILABLE' if np.isfinite(row[k]) else 'NOT_AVAILABLE' for k in METRICS},sort_keys=True)
        rows.append(row)
    return rows


def _suffix_metric(values,suffix):
    selected=[v for k,v in values.items() if k.endswith(suffix)]
    return selected[0] if len(selected)==1 else np.nan


def _probe_row(result,label):
    p=result.get('P',{}); q=result.get('Q',{})
    return dict(probe_kind='FIXED_7D_PROBE',probe_phase=label,
        fixed_parent_hash=result.get('fixed_parent_hash'),
        P0_residual=p.get('p0_norm_mae',np.nan),PI_residual=p.get('pi_norm_mae',np.nan),
        Q_residual=q.get('normalized_mae',np.nan))


def history_frames(results):
    training=[]; economic=[]; al=[]
    for sid,result in results.items():
        outer=result.get('outer'); stage=result.get('stage','NOT_AVAILABLE')
        row=dict(stage_id=sid,outer=outer,stage=stage,indices=json.dumps(result.get('indices',{}),sort_keys=True))
        for name in ('optimizer_steps','optimizer_executed','optimizer_history_rows','train_loss','dev_loss','epochs','skipped'):
            row[name]=result.get(name,np.nan)
        if stage=='cross_block': row.update(_probe_row(result,'post_macro'))
        elif stage=='data' and 'pre_macro_firm_residual' in result:
            row.update(_probe_row(result['pre_macro_firm_residual'],'pre_macro'))
        elif stage in ('P','Q'):
            row.update(probe_kind='CURRENT_FINITE_DIRECT_CONDITIONAL_BANK',
                P0_residual=result.get('full_greedy_P0_residual',np.nan),PI_residual=result.get('full_greedy_PI_residual',np.nan),
                Q_residual=result.get('fresh_Q_residual',np.nan))
        training.append(row)
        economic.extend(result.get('economic_metrics',[]))
        if stage=='SDF':
            epochs=result.get('al_history',[])
            if 'al_history_path' in result:
                path=Path(result['al_history_path'])
                if file_hash(path)!=result['al_history_sha256']: raise RuntimeError('committed SDF history changed')
                epochs=json.loads(path.read_text())
            for epoch in epochs:
                a=dict(outer=outer,chunk=result.get('indices',{}).get('chunk'),epoch=epoch['epoch'],
                    evaluation='SDF_TRUE_ONLY_FIXED_MACRO_PAIRS',dual_update=epoch['dual_update'])
                a.update(epoch.get('constraints',{})); a.update(epoch.get('al',{}))
                a['normalized_signed_aio_mean']=_suffix_metric(epoch.get('validation',{}),
                    '_primary_true_state_normalized_signed_aio_mean')
                al.append(a)
    # Stable schemas even before a stage produces any true data.
    return (pd.DataFrame(training,columns=None if training else ['outer','stage']),
            pd.DataFrame(economic,columns=None if economic else ['outer','stage','split','sampling_source','crn_hash',*METRICS]),
            pd.DataFrame(al,columns=None if al else ['outer','chunk','epoch','normalized_signed_aio_mean']))


def write_histories(root,results,*,figures=False):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    train,econ,al=history_frames(results)
    for filename,frame in [('outer_training_history.csv',train),('outer_macro_economic_history.csv',econ),('outer_sdf_al_history.csv',al)]:
        atomic_write_bytes(root/filename,frame.to_csv(index=False,na_rep='NOT_AVAILABLE').encode())
    if figures: _figures(root,train,econ,al)


def _figures(root,training,economic,al):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=root/'figures'; directory.mkdir(exist_ok=True)
    valid=economic.loc[(economic['split']=='validation')&(economic['sampling_source']=='ALL')]
    def axes(n,title):
        fig,axs=plt.subplots(1,n,figsize=(5*n,4),squeeze=False)
        fig.suptitle(title)
        for ax in axs[0]: ax.set_xlabel('Outer'); ax.set_xticks([0,1,2,3]); ax.grid(alpha=.2)
        return fig,axs[0]
    def save(fig,axs,name):
        for ax in axs:
            if not ax.lines or not any(np.isfinite(line.get_ydata()).any() for line in ax.lines):
                ax.text(.5,.5,'NOT_AVAILABLE',ha='center',transform=ax.transAxes)
            elif ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(directory/name,dpi=140); plt.close(fig)
    def macro_panel(columns,name,title):
        fig,axs=axes(len(columns),title)
        for ax,col in zip(axs,columns):
            ax.set_ylabel(col)
            for (stage,crn),g in valid.groupby(['stage','crn_hash']):
                g=g.sort_values('outer')
                ax.plot(g.outer,pd.to_numeric(g[col],errors='coerce'),marker='o',
                    linestyle='-' if stage==POST_SDF else '--',label=f'{stage} CRN:{crn[:7]}')
        save(fig,axs,name)
    macro_panel(['G_mean','G_p90'],'outer_g_residual.png','Fixed Validation CRN / ALL: normalized G (not a solved root)')
    macro_panel(['FC1_C_R2','FC1_K_R2','Delta_LnK_R2'],'outer_fc1_r2.png','Saved forecasts vs realized labels / fixed Validation CRN')
    macro_panel(['Surviving_firm_count_mean','K_mean','C_mean'],'outer_survival_capital_consumption.png','Fixed Validation CRN / ALL; post-SDF is a fresh resimulation')
    fig,axs=axes(1,'FIXED_7D_PROBE: not endogenous training residuals')
    probe=training.loc[training.get('probe_kind',pd.Series(index=training.index,dtype=object))=='FIXED_7D_PROBE']
    if not probe.empty:
        for (phase,fingerprint),g in probe.groupby(['probe_phase','fixed_parent_hash']):
            for col in ('P0_residual','PI_residual','Q_residual'):
                axs[0].plot(g.outer,pd.to_numeric(g[col],errors='coerce'),marker='o',label=f'{col} {phase} probe:{fingerprint[:7]}')
    save(fig,axs,'outer_firm_residual.png')
    fig,axs=axes(1,'SDF TRUE_ONLY: fixed-macro-pair validation normalized Signed AiO')
    if not al.empty:
        last=al.sort_values(['outer','epoch']).groupby('outer').tail(1)
        axs[0].plot(last.outer,pd.to_numeric(last.normalized_signed_aio_mean,errors='coerce'),marker='o',label='Last committed SDF epoch')
    save(fig,axs,'outer_sdf_aio.png')
