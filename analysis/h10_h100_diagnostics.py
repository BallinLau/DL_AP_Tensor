"""Columnar CPU coverage/survival reductions, no Firm forwards or plotting."""
import numpy as np
import pandas as pd
from data.finite_direct_longrun_replay import cells


def coverage_report(train,validation,coverage,replay,edges):
    result={}
    for split,f in [('train',train),('validation',validation)]:
        for axis,col in [('h','Hatc_cal_t'),('ell','LnK_cal_t')]:
            values=f[col].to_numpy(dtype=float) if len(f) else np.array([])
            for label,q in [('min',0),('p10',.1),('p50',.5),('p90',.9),('max',1)]:
                result[f'{axis}_{split}_{label}']=float(np.quantile(values,q)) if len(values) else None
        result[f'{split}_valid_rows']=len(f)
        result[f'{split}_unique_initial_lineages']=int(f.parent_candidate_id.nunique()) if len(f) else 0
    def occupied(f):return set(map(tuple,cells(f,edges))) if len(f) else set()
    selected=occupied(train);long_bins=occupied(replay);short_bins=occupied(coverage)
    inside={(h,k) for h,k in selected if 0<=h<len(edges[0])-1 and 0<=k<len(edges[1])-1}
    result.update(occupied_2d_bins=len(selected),occupied_bin_ratio=len(inside)/((len(edges[0])-1)*(len(edges[1])-1)),
        H10_visited_bin_coverage=len(selected&short_bins)/len(short_bins) if short_bins else None,
        H100_post20_visited_bin_coverage=len(selected&long_bins)/len(long_bins) if long_bins else None,
        out_of_support_valid_rows=sum(tuple(c) not in selected for c in cells(validation,edges)) if len(validation) else 0,
        FEASIBLE_SUPPORT_NARROW=len(inside)<4,
        semantics='LEGAL_TRAIN_SUPPORT_NOT_A_PROMISE_TO_FILL_RECTANGLE')
    groups=train.drop_duplicates(['replay_source','parent_candidate_id','t']) if len(train) else train
    if len(groups):
        result['longrun_share']=float((groups.replay_source=='LONGRUN').mean())
        result['coverage_share']=1-result['longrun_share']
    return result


def by_bin_errors(frame,prediction,edges,split):
    if frame.empty:return pd.DataFrame()
    f=frame[['Hatc_cal_t1','LnK_cal_t1']].reset_index(drop=True).copy()
    f['h_bin'],f['ell_bin']=cells(frame,edges).T
    for j,name in enumerate(('h','ell')):
        error=prediction[:,j]-f.iloc[:,j].to_numpy()
        f[name+'_abs']=np.abs(error);f[name+'_sq']=error*error
    out=f.groupby(['h_bin','ell_bin']).agg(rows=('h_abs','size'),h_MAE=('h_abs','mean'),
        h_MSE=('h_sq','mean'),ell_MAE=('ell_abs','mean'),ell_MSE=('ell_sq','mean')).reset_index()
    out['h_RMSE']=np.sqrt(out.pop('h_MSE'));out['ell_RMSE']=np.sqrt(out.pop('ell_MSE'));out['split']=split
    return out


def time_report(store):
    n=store.manifest['scheduled_initial_parents'];horizon=store.manifest['provenance']['horizon']
    pieces=[]
    for f in store.rows():
        if len(f):pieces.append(f)
    if not pieces:return pd.DataFrame(dict(t=np.arange(horizon+1),scheduled_paths=n,
        alive_paths=0,macro_defined_paths=0,valid_transitions=0,unique_lineages=0))
    f=pd.concat(pieces,ignore_index=True);main=f.loc[f.child_branch==0].copy()
    main['alive_path']=main.get('predecision_alive_firms',pd.Series(0,index=main.index))>0
    main['h_belief_gap']=main.Hatcf_t1-main.Hatc_cal_t1
    main['ell_belief_gap']=main.LnKF_t1-main.LnK_cal_t1
    agg=main.groupby('t').agg(alive_paths=('alive_path','sum'),macro_defined_paths=('macro_defined','sum'),
        valid_transitions=('economic_state_valid','sum'),unique_lineages=('parent_candidate_id','nunique'),
        h_belief_gap=('h_belief_gap','mean'),ell_belief_gap=('ell_belief_gap','mean'))
    for name in ('hard_default_count','entry_candidate_count','entry_accepted_count','K_cal_t1','C_raw_t1','Hatcf_t1','Hatc_cal_t1','LnKF_t1','LnK_cal_t1'):
        if name in main:agg[name]=main.groupby('t')[name].sum(min_count=1) if 'count' in name else main.groupby('t')[name].mean()
    out=agg.reindex(np.arange(horizon+1));out.index.name='t'
    for name in ('alive_paths','macro_defined_paths','valid_transitions','unique_lineages'):out[name]=out[name].fillna(0)
    # Initial observations are separate real nodes, not synthetic transitions.
    out.loc[0,:]=np.nan
    initial=store.root/'initial_observations.csv'
    if initial.exists():
        from data.finite_direct_sampling import file_hash
        if file_hash(initial)!=store.manifest['initial_observations_sha256']:raise ValueError('initial diagnostic corruption')
        first=pd.read_csv(initial)
        if len(first):
            out.loc[0,'alive_paths']=(first.predecision_alive_firms>0).sum()
            out.loc[0,'macro_defined_paths']=first.macro_defined.sum()
            out.loc[0,'unique_lineages']=first.parent_candidate_id.nunique()
            for name in ('hard_default_count','entry_candidate_count','entry_accepted_count','K_cal_t1','C_raw_t1','Hatcf_t1','Hatc_cal_t1','LnKF_t1','LnK_cal_t1'):
                if name in first:out.loc[0,name]=first[name].sum(min_count=1) if 'count' in name else first[name].mean()
    out['scheduled_paths']=n
    return out.reset_index()
