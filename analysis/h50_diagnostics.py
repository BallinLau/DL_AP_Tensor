"""CPU streaming coverage and error plots, never neural forwards/optimizers."""
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.nested_equilibrium_utils import atomic_write_json


def streaming_economic_metrics(store,schedule,*,outer,stage):
    """All scheduled H50 rows, not the bounded optimizer subset. Only G scalar
    arrays are retained for exact quantiles; other metrics use sufficient stats.
    """
    from analysis.finite_direct_outer_metrics import METRICS
    from analysis.nested_equilibrium_utils import stable_json_hash
    source={r['parent_id']:r['sampling_source'] for r in schedule['parents']}
    accum={g:dict(sums=defaultdict(float),counts=defaultdict(int),g=[],
        valid=0,pairs=0,attempts=0,parent_attempts=0,parent_valid=0,lineages=set()) for g in ('BETA','EMPIRICAL','ALL')}
    for shard in store.chunks():
        frame=pd.DataFrame(shard['rows'])
        for group,a in accum.items():
            parents=[r for r in shard.get('parent_records',[]) if group=='ALL' or source.get(r['parent_candidate_id'])==group]
            a['parent_attempts']+=len(parents);a['parent_valid']+=sum(r['economic_state_valid'] for r in parents)
            f=frame if group=='ALL' else frame[frame.parent_candidate_id.map(source)==group]
            valid=f[f.economic_state_valid.astype(bool)]
            a['valid']+=len(valid);a['pairs']+=int((valid.groupby(['parent_candidate_id','t']).size()==2).sum())
            a['attempts']+=int(f.get('node_attempted',pd.Series(False,index=f.index)).sum())
            a['lineages'].update(valid.parent_candidate_id.tolist())
            for name in ('surviving_mass','K_cal_t1','C_raw_t1','default_mass','investment_mass','entry_mass','firm_mass'):
                if name in f:
                    values=pd.to_numeric(f[name],errors='coerce').to_numpy();values=values[np.isfinite(values)]
                    a['sums'][name]+=float(values.sum());a['counts'][name]+=len(values)
            if not len(valid):continue
            if 'normalized_G_at_prediction' in valid:
                values=valid.normalized_G_at_prediction.to_numpy();a['g'].append(values[np.isfinite(values)])
            for name,y,pred in [('FC1_C',valid.Hatc_cal_t1.to_numpy(),valid.Hatcf_t1.to_numpy()),
                                ('FC1_K',valid.LnK_cal_t1.to_numpy(),valid.LnKF_t1.to_numpy()),
                                ('Delta_LnK',(valid.LnK_cal_t1-valid.LnK_cal_t).to_numpy(),(valid.LnKF_t1-valid.LnK_cal_t).to_numpy())]:
                ok=np.isfinite(y)&np.isfinite(pred);y=y[ok];pred=pred[ok]
                a['sums'][name+'_y']+=float(y.sum());a['sums'][name+'_y2']+=float((y*y).sum())
                a['sums'][name+'_sse']+=float(((pred-y)**2).sum());a['sums'][name+'_abs']+=float(np.abs(pred-y).sum())
                a['counts'][name]+=len(y)
    rows=[]
    for group,a in accum.items():
        count=sum(group=='ALL' or label==group for label in source.values());den=count*50
        row={name:float('nan') for name in METRICS}
        row.update(outer=outer,stage=stage,split=schedule['split'],sampling_source=group,
            crn_hash=stable_json_hash([r['child_shock_keys'] for r in schedule['parents']]),
            scheduled_parent_count=count,scheduled_parent_time_count=den,scheduled_child_count=2*den,
            attempted_child_count=a['attempts'],valid_child_count=a['valid'],complete_sdf_pair_count=a['pairs'],
            attempted_parent_time_count=a['parent_attempts'],valid_parent_time_count=a['parent_valid'],
            Valid_child_fraction=a['valid']/(2*den) if den else np.nan,
            Complete_SDF_pair_fraction=a['pairs']/den if den else np.nan,
            metric_scope='ALL_SCHEDULED_H50_ROWS; INVALID_SUFFIXES_RETAINED',
            empirical_semantics='TRAIN_DYNAMIC_PRE_DECISION_REALIZED_B; FIXED_INITIAL_VALIDATION')
        g=np.concatenate(a['g']) if a['g'] else np.array([])
        if len(g):row.update(G_mean=float(g.mean()),G_median=float(np.median(g)),G_p90=float(np.quantile(g,.9)),G_p99=float(np.quantile(g,.99)))
        for out,key in [('Surviving_firm_count_mean','surviving_mass'),('K_mean','K_cal_t1'),('C_mean','C_raw_t1')]:
            row[out]=a['sums'][key]/a['counts'][key] if a['counts'][key] else np.nan
        for out,key in [('Default_rate','default_mass'),('Investment_rate','investment_mass'),('Entry_rate','entry_mass')]:
            row[out]=a['sums'][key]/a['sums']['firm_mass'] if a['sums']['firm_mass'] else np.nan
        for name in ('FC1_C','FC1_K','Delta_LnK'):
            n=a['counts'][name];denom=a['sums'][name+'_y2']-a['sums'][name+'_y']**2/n if n else 0
            row[name+'_R2']=1-a['sums'][name+'_sse']/denom if denom>0 else np.nan
        for out,name in [('Hatc_forecast_MAE','FC1_C'),('LnK_forecast_MAE','FC1_K')]:
            row[out]=a['sums'][name+'_abs']/a['counts'][name] if a['counts'][name] else np.nan
        rows.append(row)
    return rows


def write_diagnostics(store, output):
    if store.manifest.get('provenance',{}).get('diagnostics_version')=='H50_JOINT_COVERAGE_DIAGNOSTICS_V1':
        from analysis.h50_joint_diagnostics import write_dashboard
        from data.finite_direct_joint_coverage import JointCoverageConfig
        cfg=store.manifest['provenance'].get('diagnostic_metadata',{}).get('coverage_config',{})
        return write_dashboard(store,output,JointCoverageConfig(**cfg))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    edges=[np.linspace(-15,8,93),np.linspace(-12,5,69)]
    names=['fc1_h_lnk_coverage_heatmap','fc1_long_horizon_actual_states',
           'firm_onpolicy_macro_coverage','firm_calculated_macro_coverage']
    hist={name:np.zeros((92,68)) for name in names};by_time=defaultdict(lambda:defaultdict(float))
    time_hist=np.zeros((3,92,68));attempted_hist=np.zeros((92,68))
    omitted=defaultdict(int);errors=np.zeros((92,68));error_counts=np.zeros_like(errors)
    def add(name,k,h):
        values=np.histogram2d(k,h,bins=edges)[0];hist[name]+=values
        omitted[name]+=len(k)-int(values.sum())
    for shard in store.chunks():
        rows=pd.DataFrame(shard['rows'])
        if {'LnK_cal_t','Hatc_cal_t'} <= set(rows):
            known=rows[['LnK_cal_t','Hatc_cal_t']].to_numpy(dtype=float)
            known=known[np.isfinite(known).all(axis=1)]
            attempted_hist+=np.histogram2d(known[:,0],known[:,1],bins=edges)[0]
        for t,g in rows.groupby('t'):
            v=g.loc[g.economic_state_valid.astype(bool)];stats=by_time[int(t)]
            parents=shard.get('parent_records',[])
            stats['parent_nodes_observed']+=len(parents)
            stats['valid_parent_nodes']+=sum(r['economic_state_valid'] for r in parents)
            stats['attempted_child_rows']+=len(g);stats['valid_child_rows']+=len(v)
            stats['complete_sdf_pairs']+=int((g.groupby('parent_candidate_id').economic_state_valid.sum()==2).sum())
            if len(v):
                add(names[0],v.LnK_cal_t.to_numpy(),v.Hatc_cal_t.to_numpy())
                band=0 if int(t)-1<10 else 1 if int(t)-1<30 else 2
                time_hist[band]+=np.histogram2d(v.LnK_cal_t,v.Hatc_cal_t,bins=edges)[0]
                add(names[1],v.LnK_cal_t1.to_numpy(),v.Hatc_cal_t1.to_numpy())
                stats['Hatc_forecast_abs_error_sum']+=float((v.Hatcf_t1-v.Hatc_cal_t1).abs().sum())
                stats['LnK_forecast_abs_error_sum']+=float((v.LnKF_t1-v.LnK_cal_t1).abs().sum())
                for field in ('surviving_mass','default_mass','entry_mass','K_cal_t1','C_raw_t1'):
                    if field in v:stats[field+'_sum']+=float(v[field].sum())
                points=[v.LnK_cal_t.to_numpy(),v.Hatc_cal_t.to_numpy()]
                errors+=np.histogram2d(*points,bins=edges,weights=(v.Hatcf_t1-v.Hatc_cal_t1).abs())[0]
                error_counts+=np.histogram2d(*points,bins=edges)[0]
        for c in shard['captures']:
            m=c['eligible'].numpy().astype(bool);s=c['states'].numpy()[m];context=c['context'].numpy()[m]
            add(names[2],s[:,6],s[:,5]);add(names[3],context[:,1],context[:,0])
    frame=pd.DataFrame([dict(t=t,**v) for t,v in sorted(by_time.items())])
    frame['valid_child_fraction']=frame.valid_child_rows/frame.attempted_child_rows
    scheduled=store.manifest.get('scheduled_initial_parents')
    frame['valid_parent_fraction']=frame.valid_parent_nodes/scheduled if scheduled else np.nan
    for field in ('Hatc','LnK'):
        frame[field+'_forecast_MAE']=frame[field+'_forecast_abs_error_sum']/frame.valid_child_rows.replace(0,np.nan) if field+'_forecast_abs_error_sum' in frame else np.nan
    frame.to_csv(output/'coverage_and_error_by_time.csv',index=False)
    for name,values in hist.items():
        fig,ax=plt.subplots(figsize=(7,5));mesh=ax.pcolormesh(edges[0],edges[1],np.log1p(values.T),shading='auto')
        ax.set(xlabel='LnK',ylabel='Hatc',title=name+' (log1p count)')
        fig.colorbar(mesh,ax=ax);fig.tight_layout();fig.savefig(output/(name+'.png'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));mean=np.divide(errors,error_counts,out=np.full_like(errors,np.nan),where=error_counts>0)
    mesh=ax.pcolormesh(edges[0],edges[1],mean.T,shading='auto');fig.colorbar(mesh,ax=ax)
    ax.set(xlabel='Parent LnK_cal',ylabel='Parent Hatc_cal',title='FC1 Hatc absolute prediction error')
    fig.tight_layout();fig.savefig(output/'fc1_error_macro_heatmap.png');plt.close(fig)
    frame.plot(x='t',y='valid_parent_fraction',ylim=(0,1),legend=False,title='Valid main parents / all scheduled initial lineages (parent time=t-1)')
    plt.tight_layout();plt.savefig(output/'valid_parent_fraction_by_time.png');plt.close()
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,values,label in zip(axes,time_hist,['Parent t=0..9','Parent t=10..29','Parent t=30..49']):
        ax.pcolormesh(edges[0],edges[1],np.log1p(values.T),shading='auto')
        ax.set(xlabel='Parent LnK_cal',ylabel='Parent Hatc_cal',title=label)
    fig.suptitle('Actual eligible input coverage; empty bins are not fabricated')
    fig.tight_layout();fig.savefig(output/'fc1_h_lnk_coverage_by_time.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,col in zip(axes,['surviving_mass_sum','entry_mass_sum']):
        if col in frame:
            ax.plot(frame.t,frame[col]/frame.valid_child_rows.replace(0,np.nan))
        ax.set(xlabel='Child t',ylabel=col,title='Conditional on valid children')
    fig.tight_layout();fig.savefig(output/'survival_and_entry_by_time.png');plt.close(fig)
    valid_fraction=np.divide(hist[names[0]],attempted_hist,out=np.full_like(attempted_hist,np.nan),where=attempted_hist>0)
    x,y=np.meshgrid(edges[0][:-1],edges[1][:-1],indexing='ij')
    pd.DataFrame(dict(lnK_lower=x.ravel(),h_lower=y.ravel(),attempted=attempted_hist.ravel(),
        valid=hist[names[0]].ravel(),valid_fraction=valid_fraction.ravel(),
        forecast_abs_error=mean.ravel())).to_csv(output/'fc1_macro_bins.csv',index=False)
    atomic_write_json(output/'diagnostic_status.json',dict(omitted_histogram_observations=omitted,
        firm_bellman_residual='POPULATION_NOT_AVAILABLE; SEE_SEPARATE_SMALL_FORWARD_PROBE_IF_PROVIDED',
        counts='FC1 input count is child-weighted; not independent parent count',
        source='ACTUAL_STORED_LEDGER_ROWS',training_performed=False))
    return frame


def write_bellman_heatmap(points,output):
    """Plot supplied REAL per-row residuals; never infer them from loss means."""
    required={'lnkf','hatcf','p0_normalized','pi_normalized'}
    if not required <= set(points):raise ValueError('real per-row Bellman residuals required')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    x=points.lnkf.to_numpy();y=points.hatcf.to_numpy()
    residual=.5*(points.p0_normalized.abs()+points.pi_normalized.abs()).to_numpy()
    if not (np.isfinite(x)&np.isfinite(y)&np.isfinite(residual)).all():
        raise ValueError('nonfinite Bellman heatmap observations')
    count,xe,ye=np.histogram2d(x,y,bins=12)
    total=np.histogram2d(x,y,bins=[xe,ye],weights=residual)[0]
    mean=np.divide(total,count,out=np.full_like(total,np.nan),where=count>0)
    fig,ax=plt.subplots(figsize=(7,5));mesh=ax.pcolormesh(xe,ye,mean.T,shading='auto');fig.colorbar(mesh,ax=ax)
    ax.set(xlabel='Firm input LnKF',ylabel='Firm input Hatcf',title=f'Forward-only sampled Bellman residual (n={len(points)}); NOT population')
    fig.tight_layout();fig.savefig(output);plt.close(fig)
