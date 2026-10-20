"""CPU aggregation of immutable banks; no model forward or optimizer."""
import numpy as np
import pandas as pd
from data.finite_direct_dynamic_empirical import load_bundle


def write_dynamic_histories(backend,*,figures=False):
    root=backend.root
    pool_rows=list(backend.state.get('dynamic_history',{}).values())
    pd.DataFrame(pool_rows).to_csv(root/'empirical_pool_history.csv',index=False)
    rows=[];hist=[]
    for outer,record in sorted(backend.state.get('dynamic_banks',{}).items(),key=lambda kv:int(kv[0])):
        pool,bank,schedule=load_bundle(record)
        b=bank['state']['b'].numpy();k=bank['state']['K'].numpy();mass=bank['state']['alive'].numpy()
        for kind in ('BETA','EMPIRICAL'):
            select=np.array([p['sampling_source']==kind for p in schedule['parents']]);mask=mass[select]
            values=b[select][mask];capital=k[select][mask]
            rows.append(dict(outer=int(outer),sampling_source=kind,split='train',
                evaluation_distribution='CHANGING_TRAIN_NOT_FIXED_VALIDATION',
                parent_count=int(select.sum()),b_mean=float(values.mean()),b_std=float(values.std()),
                b_p10=float(np.quantile(values,.1)),b_p50=float(np.quantile(values,.5)),b_p90=float(np.quantile(values,.9)),
                high_b_fraction=float((values>.5).mean()),capital_weighted_b_mean=float((values*capital).sum()/capital.sum()),
                b_capital_correlation=float(np.corrcoef(values,capital)[0,1]) if values.std()>0 and capital.std()>0 else None,
                pool_fingerprint=pool['fingerprint'],train_bank_fingerprint=bank['fingerprint'],
                template_count=len(pool['recent'])+len(pool['archive']),
                recent_template_count=len(pool['recent']),archive_template_count=len(pool['archive']),
                crn_manifest_hash=record['crn_manifest_hash'],
                initial_empirical_semantics='LOCKED_JOB699894_T0' if int(outer)==4 else 'DYNAMIC_TRAIN_REALIZED_B'))
        hist.extend(record.get('history',[]))
    frame=pd.DataFrame(rows);frame.to_csv(root/'train_distribution_history.csv',index=False)
    pd.DataFrame(hist).to_csv(root/'empirical_parent_sampling_history.csv',index=False)
    if not figures or frame.empty:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=root/'figures';target.mkdir(exist_ok=True)
    def plot(name,columns,title):
        fig,ax=plt.subplots(figsize=(8,4))
        for source,g in frame.groupby('sampling_source'):
            for col in columns: ax.plot(g.outer,g[col],marker='o',label=f'{source} {col}')
        ax.set(xlabel='Outer initial Train Bank (not fixed Validation)',title=title)
        ax.set_xticks(sorted(frame.outer.unique()));ax.legend();fig.tight_layout();fig.savefig(target/name);plt.close(fig)
    plot('empirical_b_mean_p90_by_outer.png',['b_mean','b_p90'],'Train b mean / P90; mixture unchanged')
    plot('beta_vs_empirical_b_distribution.png',['b_p10','b_p50','b_p90'],'Train b quantiles; Beta design support unchanged')
    plot('empirical_template_coverage.png',['recent_template_count','archive_template_count'],'Pool template counts (not independent Parent counts)')
    plot('empirical_b_distribution_by_outer.png',['b_p10','b_p50','b_p90'],'Initial Train distribution quantiles by Outer')
