"""Forward-only fixed-input diagnostics; never optimize or change probe inputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from analysis.nested_equilibrium_utils import atomic_torch_save, capture_rng_state,restore_rng_state
from analysis.finite_direct_continuation_state import state_digest,safe_load,component_hashes
from data.finite_direct_macro_contract import production_sdf_tensors


def movement(current, previous, older=None):
    x=np.asarray(current,dtype=float).reshape(-1); y=np.asarray(previous,dtype=float).reshape(-1)
    if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite or mismatched fixed probe')
    delta=x-y; absolute=np.abs(delta)
    result=dict(rms=float(np.sqrt(np.mean(delta**2))),mae=float(absolute.mean()),
        p90=float(np.quantile(absolute,.9)),p99=float(np.quantile(absolute,.99)))
    if older is not None:
        z=np.asarray(older,dtype=float).reshape(-1)
        if z.shape!=x.shape or not np.isfinite(z).all(): raise ValueError('bad two-Outer probe')
        old_delta=y-z; denom=np.linalg.norm(delta)*np.linalg.norm(old_delta)
        result.update(two_outer_rms=float(np.sqrt(np.mean((x-z)**2))),
                      update_direction_cosine=float(delta@old_delta/denom) if denom>0 else np.nan)
    return result


def make_macro_probe(frame):
    tensors=production_sdf_tensors(frame,device='cpu')
    # All complete Validation parents, not independent firm mass observations.
    return dict(parent=tensors['parents'],children=torch.stack(tensors['children'],dim=1),
                lineage=tensors['lineage'])


@torch.no_grad()
def evaluate_fixed(backend, firm_probe, macro_probe):
    models=backend.models; device=backend.episode.device
    rng=capture_rng_state(); modes={m:m.training for model in models.values() for m in model.modules()}
    before=state_digest({k:v.state_dict() for k,v in models.items()})
    try:
        for model in models.values(): model.eval()
        s=firm_probe.parent_states.to(device); out=models['policy_value'](s)
        firm={k:getattr(out,k).detach().cpu().reshape(-1) for k in ('P0','PI','Q','bp0','bpI')}
        firm['default']=(out.Phat<=0).cpu().reshape(-1)
        firm['investment']=((out.PI>=out.P0)&(out.Phat>0)).cpu().reshape(-1)
        p=macro_probe['parent'].to(device); c=macro_probe['children'].to(device)
        macro=models['sdf_fc1']; inputs=torch.stack([p[:,4,None].expand(-1,2),c[:,:,4],
            p[:,7,None].expand(-1,2),p[:,8,None].expand(-1,2)],dim=-1)
        fc=macro.forward_fc1(inputs.reshape(-1,4),return_physical=True)
        wp,wc,m,_,_=macro.forward_step(p[:,4:5],c[:,:,4:5],p[:,7:8],p[:,8:9],return_physical=True)
        used=m.clamp(backend.episode.hyperparams.pv_m_clamp_min,backend.episode.hyperparams.pv_m_clamp_max)
        # Also evaluate TRUE-state wealth/M, distinct from the FC1 perceived
        # transition that determines the actual Firm conditional pricing kernel.
        wt=macro.value_model(torch.stack([p[:,4],p[:,7],p[:,8]],dim=-1))
        wct=macro.value_model(torch.stack([c[:,:,4],c[:,:,8],c[:,:,9]],dim=-1).reshape(-1,3)).reshape(len(p),2)
        mt=macro.sdf_model.get_M(wt,wct,p[:,8],c[:,:,9],p[:,7],c[:,:,8])
        if isinstance(mt,list): mt=torch.stack(mt,dim=1)
        result=dict(firm=firm,fc1=dict(FC1_C=fc[0].cpu().reshape(-1),FC1_K=fc[1].cpu().reshape(-1)),
            sdf=dict(W_parent=wp.cpu().reshape(-1),W_child=wc.cpu().reshape(-1),
                     M_raw=m.cpu().reshape(-1),M_used=used.cpu().reshape(-1),
                     W_true_parent=wt.cpu().reshape(-1),W_true_child=wct.cpu().reshape(-1),M_true_raw=mt.cpu().reshape(-1)),
            probe_fingerprint=state_digest(dict(firm=firm_probe.payload(),macro=macro_probe)),
            model_hashes=component_hashes({k:v.state_dict() for k,v in models.items()}),
            pricing_context='FIXED_TRUE_PARENT_CURRENT_FC1_PERCEIVED_CHILD',
            bp_context='BP_HEAD_FUNCTION; actual Simulation BP remains production Grid')
        if any(not torch.isfinite(t).all() for group in ('firm','fc1','sdf') for t in result[group].values()):
            raise FloatingPointError('nonfinite fixed-probe output')
    finally:
        for mod,mode in modes.items(): mod.training=mode
        restore_rng_state(rng)
    if state_digest({k:v.state_dict() for k,v in models.items()})!=before: raise RuntimeError('probe mutated model')
    return result


def save_movement(root, outer, current, previous=None, older=None):
    root=Path(root)
    if previous is not None and current['probe_fingerprint']!=previous['probe_fingerprint']:
        raise ValueError('fixed probe identity changed')
    if older is not None and current['probe_fingerprint']!=older['probe_fingerprint']:
        raise ValueError('two-Outer fixed probe identity changed')
    rows={group:[] for group in ('firm','fc1','sdf')}
    for group in rows:
        for name,x in current[group].items():
            numeric=x.double().numpy()
            row=dict(outer=outer,function=name,probe_kind='FIXED_INPUT_PROBE',
                probe_fingerprint=current['probe_fingerprint'],mean=float(numeric.mean()),
                variance=float(numeric.var()),p90_value=float(np.quantile(numeric,.9)),
                p99_value=float(np.quantile(numeric,.99)),baseline=previous is None)
            row['model_hashes']=str(current.get('model_hashes',{}))
            if previous is not None:
                row.update(movement(numeric,previous[group][name].double().numpy(),
                    older[group][name].double().numpy() if older is not None else None))
                if name in ('default','investment'): row['flip_fraction']=float((x!=previous[group][name]).double().mean())
            if name=='M_used': row['clamp_fraction']=float((x!=current['sdf']['M_raw']).double().mean())
            rows[group].append(row)
    return rows


FIGURES={
 'firm_P0_function_movement.png':('firm','P0','rms'),
 'firm_PI_function_movement.png':('firm','PI','rms'),
 'firm_Q_function_movement.png':('firm','Q','rms'),
 'firm_BP_function_movement.png':('firm','bp0','rms'),
 'firm_default_investment_flip.png':('firm','default','flip_fraction'),
 'fc1_C_function_movement.png':('fc1','FC1_C','rms'),
 'fc1_K_function_movement.png':('fc1','FC1_K','rms'),
 'wealth_function_movement.png':('sdf','W_parent','rms'),
 'M_raw_function_movement.png':('sdf','M_raw','rms'),
 'M_used_function_movement.png':('sdf','M_used','rms'),
 'M_distribution_by_outer.png':('sdf','M_raw','mean'),
 'cross_block_function_movement.png':('firm','Q','rms'),
 'outer_fixed_point_summary.png':('firm','P0','two_outer_rms')}


def write_probe_histories(root, results, *, figures=False):
    root=Path(root)
    names=dict(firm='firm_function_movement.csv',fc1='fc1_function_movement.csv',sdf='sdf_wealth_function_movement.csv')
    frames={}
    for group,name in names.items():
        rows=[row for result in results.values() for row in result.get('function_movement',{}).get(group,[])]
        frames[group]=pd.DataFrame(rows)
        frames[group].to_csv(root/name,index=False)
    timing=pd.DataFrame([dict(outer=r['outer'],stage=r['stage'],seconds=r['stage_wallclock_seconds'])
        for r in results.values() if 'stage_wallclock_seconds' in r])
    timing.to_csv(root/'outer_execution_timing.csv',index=False)
    if not figures: return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=root/'figures';target.mkdir(exist_ok=True)
    def plot(frame, filename, column, label):
        columns=[column] if isinstance(column,str) else column
        if frame.empty or not any(c in frame for c in columns): return
        fig,ax=plt.subplots(figsize=(7,4))
        for col in columns:
            if col in frame: ax.plot(frame.outer,frame[col],marker='o',label=col)
        ax.set(xlabel='Outer (3 = NEW BANK BASELINE)',title=label)
        ax.set_xticks(sorted(frame.outer.unique()));ax.legend()
        fig.tight_layout();fig.savefig(target/filename);plt.close(fig)
    for filename,(group,name,col) in FIGURES.items():
        frame=frames[group]
        if not frame.empty:
            subset=frame[frame.function==name]
            if filename=='firm_BP_function_movement.png' and 'rms' in frame:
                subset=frame[frame.function.isin(['bp0','bpI'])].pivot(index='outer',columns='function',values='rms').reset_index()
                plot(subset,filename,['bp0','bpI'],'Fixed-input BP HEAD function movement (not Grid action)')
            elif filename=='firm_default_investment_flip.png' and 'flip_fraction' in frame:
                subset=frame[frame.function.isin(['default','investment'])].pivot(index='outer',columns='function',values='flip_fraction').reset_index()
                plot(subset,filename,['default','investment'],'Fixed-input hard decision flips')
            elif filename=='M_distribution_by_outer.png':
                plot(subset,filename,['mean','variance','p90_value','p99_value'],'Raw M on fixed parents/shocks')
            else: plot(subset,filename,col,name+' fixed inputs')
    if not timing.empty: plot(timing.groupby('outer',as_index=False).seconds.sum(),'outer_training_wallclock.png','seconds','Stage wallclock')
    economic_path=root/'outer_macro_economic_history.csv'
    if economic_path.exists():
        f=pd.read_csv(economic_path)
        if len(f):
            f=f[(f.split=='validation')&(f.sampling_source=='ALL')&f.stage.isin(
                ['POST_SDF_VALIDATION_RESIMULATION','OUTER3_ON_NEW_VALIDATION_BANK_BASELINE'])]
            for filename,col in {'fc1_C_R2_by_outer.png':'FC1_C_R2','fc1_K_R2_by_outer.png':'FC1_K_R2',
                'fc1_delta_lnk_R2_by_outer.png':'Delta_LnK_R2','macro_G_mean_P90_P99_by_outer.png':['G_mean','G_p90','G_p99'],
                'surviving_firms_by_outer.png':'Surviving_firm_count_mean','default_investment_entry_by_outer.png':['Default_rate','Investment_rate','Entry_rate'],
                'capital_by_outer.png':'K_mean','consumption_by_outer.png':'C_mean',
                'economic_validity_coverage.png':'Valid_child_fraction'}.items(): plot(f,filename,col,'Fixed new Validation CRN')
    al_path=root/'outer_sdf_al_history.csv'
    if al_path.exists():
        al=pd.read_csv(al_path)
        if len(al):
            last=al.sort_values(['outer','epoch']).groupby('outer').tail(1)
            plot(last,'SDF_AiO_by_outer.png','normalized_signed_aio_mean','Last committed TRUE_ONLY epoch')
            plot(last,'SDF_AL_moment_constraints.png',['max_violation','lambda_mean_low','lambda_mean_high','lambda_var_high','rho'],'Signed AiO AL state')
    residual=root/'firm_residual_history.csv'
    if residual.exists():
        r=pd.read_csv(residual)
        if len(r): plot(r[r.probe_kind=='FIXED_7D_PROBE'],'firm_bellman_residual_by_outer.png',
            ['P0_residual','PI_residual','Q_residual'],'Fixed 7D probe / current Macro')
