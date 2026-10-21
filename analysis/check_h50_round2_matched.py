"""Bounded CPU matched Firm inference, freshly rebuilt RHS and independent roots.

Never invokes a trainer. A/B checkpoints are immutable inputs; missing B is a
reported blocker, not permission to train a substitute.
"""
import argparse,json,time,resource,platform,sys,subprocess
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.finite_direct_continuation_state import safe_load,component_hashes,state_digest
from analysis.nested_equilibrium_utils import atomic_write_json,stable_json_hash
from analysis.run_fc1_self_consistent_macro import setup,hashes,select_state
from analysis.h50_joint_diagnostics import write_dashboard
from analysis.economic_config import AnalysisEconomicConfig
from data.finite_direct_runtime import conditional_child_bank,firm_bp_parent_rows
from data.finite_direct_long_horizon_runtime import generate_long_horizon_dataset
from data.finite_direct_long_horizon_empirical import committed_templates
from data.finite_direct_joint_coverage import JointCoverageConfig,eligible_templates
from data.finite_direct_sampling import file_hash,state_hash
from training.inner_fixed_point import InnerFPCollocationBank,evaluate_inner_operator_snapshot
from training.bp_simulation_policy import GridBPSimulationPolicy
from config import Config


def matched_payloads(before,after):
    a=component_hashes(before['transaction']['models']);b=component_hashes(after['transaction']['models'])
    if any(a[k]!=b[k] for k in a if k!='firm'):raise ValueError('UNMATCHED_MACRO_CHECKPOINTS')
    if state_digest(before['transaction']['models']['sdf_fc1'])!=state_digest(after['transaction']['models']['sdf_fc1']):raise ValueError('macro running state drift')
    return a,b


def static_grid(extra_contexts=()):
    # Fixed macro cases; the same i/x and eta slices are used for A and B.
    macros=[(float(Config.XBAR),h,k) for h,k in [(-1.,4.19),(-6.,-2.),(-6.,5.),(2.,-2.),(2.,5.),(-1.879,-2.156),(-.9,1.),(-.5,0.)]]+list(extra_contexts)
    mesh=torch.cartesian_prod(torch.linspace(0,1,9),torch.linspace(-2,2,9),torch.tensor([0.,1.]),torch.arange(len(macros)).float())
    s=torch.empty(len(mesh),7);s[:,:3]=mesh[:,:3];s[:,3]=.25;s[:,4]=float(Config.XBAR)
    s[:,4:7]=torch.tensor(macros)[mesh[:,3].long()]
    b=InnerFPCollocationBank(s,torch.full((len(s),),2),2026102142,'validation',dict(
        macro_context_kind='SYNTHETIC_CURRENT_MACRO_PERCEIVED_FC1_TRANSITION',transition_macro_context=s[:,5:7].clone(),
        source_kind=['FIRM_UNIFORM_MACRO_COLLOCATION']*len(s),macro_case=mesh[:,3].long(),population_distribution=False))
    return b


def evaluate_static(models,e,bank,out):
    macro_before=state_digest(models['sdf_fc1'].state_dict())
    child=conditional_child_bank(models['sdf_fc1'],bank,seed=77,n_child_shocks=2,economic=AnalysisEconomicConfig.from_current_config(),device='cpu',chunk_size=128,
        m_min=e.hyperparams.pv_m_clamp_min,m_max=e.hyperparams.pv_m_clamp_max)
    parent=firm_bp_parent_rows(bank);batches=[dict(parent=parent[j:j+128],children=list(child.children[j:j+128].unbind(1))) for j in range(0,len(parent),128)]
    metrics,policies,snapshot=evaluate_inner_operator_snapshot(e,batches,include_bp_regret=True)
    _,_,reference_snapshot=evaluate_inner_operator_snapshot(e,batches[:1],include_bp_regret=False)
    target_parity={k:float((snapshot[k][:len(reference_snapshot[k])]-reference_snapshot[k]).abs().max()) for k in ('p0_target','pi_target','q_target')}
    for k,error in target_parity.items():
        scale=max(1.,float(reference_snapshot[k].abs().max()))
        if error>2e-6*scale:raise ValueError('DIAGNOSTIC_GRID_TARGET_PARITY_FAILED')
    model=models['policy_value'];output=model(bank.parent_states)
    frame=pd.DataFrame(bank.parent_states.numpy(),columns=['b','z','eta','i','x','h_f','ell_f'])
    frame['macro_case']=bank.metadata['macro_case'].numpy()
    for name in ('P0','PI','Phat','Q','bp0','bpI','bar_i','bar_z'):frame[name]=getattr(output,name).reshape(-1).detach().numpy()
    frame['hard_default']=frame.bar_z>=.5
    frame['hard_investment_at_fixed_i']=frame.PI>=frame.P0
    # Cost-region diagnostic, not a new threshold/decision rule. Resolve all
    # observed PI-P0 sign changes on the ORIGINAL production i grid.
    igrid=torch.linspace(0.,model.i_threshold,max(2,model.i_grid_size))
    costs=bank.parent_states[:,None,:].expand(-1,len(igrid),-1).clone();costs[:,:,3]=igrid[None,:]
    flat=costs.reshape(-1,7);parts=[]
    for start in range(0,len(flat),2048):
        p0,pi=model._value_outputs(flat[start:start+2048]);parts.append((pi-p0).reshape(-1))
    diff=torch.cat(parts).reshape(len(bank.parent_states),-1);active=diff>=0
    crossings=active[:,:-1]!=active[:,1:];rid,jid=torch.nonzero(crossings,as_tuple=True)
    pd.DataFrame(dict(state_index=rid.numpy(),i_left=igrid[jid].numpy(),i_right=igrid[jid+1].numpy(),
        difference_left=diff[rid,jid].numpy(),difference_right=diff[rid,jid+1].numpy())).to_csv(out/'investment_threshold_brackets.csv',index=False)
    frame['investment_cost_grid_crossings']=crossings.sum(1).numpy()
    frame['investment_cost_grid_positive_share']=active.float().mean(1).numpy()
    for k,v in snapshot.items():
        if len(v)==len(frame):frame[k]=v.reshape(-1).numpy()
    for k,v in policies.items():frame[k]=v.reshape(-1).numpy()
    for v in ('p0','pi','q'):frame[v+'_residual']=frame[v+'_value']-frame[v+'_target']
    frame['bp0_MAE']=(frame.bp0-frame.b0_star).abs();frame['bpi_MAE']=(frame.bpI-frame.bi_star).abs()
    frame['bp_execution_applicable']=frame.eta==1
    frame.loc[frame.eta==0,['bp0_MAE','bpi_MAE','bp0_regret','bpi_regret']]=np.nan
    frame.to_csv(out/'static_matched_states.csv',index=False)
    frame['h_bin']=pd.cut(frame.h_f,[-float('inf'),-2,0,2,float('inf')],labels=False)
    frame['ell_bin']=pd.cut(frame.ell_f,[-float('inf'),0,2,5,float('inf')],labels=False)
    frame['b_bin']=pd.cut(frame.b,[-.001,.25,.5,.75,1.001],labels=False)
    frame['z_bin']=pd.cut(frame.z,[-float('inf'),-1,0,1,float('inf')],labels=False)
    summaries=[]
    for field in ('eta','h_bin','ell_bin','b_bin','z_bin'):
        for value,g in frame.groupby(field):
            summaries.append(dict(group=field,value=int(value),n=len(g),default_area=float(g.hard_default.mean()),
                **{k+'_MAE':float(g[k].abs().mean()) for k in ('p0_residual','pi_residual','q_residual','bp0_regret','bpi_regret','bp0_MAE','bpi_MAE')}))
    pd.DataFrame(summaries).to_csv(out/'static_residual_groups.csv',index=False)
    boundary=[]
    for (case,eta),g in frame.groupby(['macro_case','eta']):
        m=g.pivot(index='b',columns='z',values='hard_default').to_numpy(dtype=int)
        crossings=np.abs(np.diff(m,axis=0)).sum(0)
        boundary.append(dict(macro_case=int(case),eta=int(eta),default_area=float(m.mean()),
            b_direction_reversals=int((np.diff(m,axis=0)<0).sum()),z_direction_reversals=int((np.diff(m,axis=1)>0).sum()),
            multiple_b_crossing_columns=int((crossings>1).sum()),boundary_total_variation=float(np.abs(np.diff(m,axis=0)).sum()+np.abs(np.diff(m,axis=1)).sum()),
            value_abs_max=float(g[['P0','PI','Q']].abs().to_numpy().max())))
    pd.DataFrame(boundary).to_csv(out/'boundary_quality.csv',index=False)
    raw=child.metadata['raw_m'];used=child.children[:,:,7]
    atomic_write_json(out/'static_contract.json',dict(parent_fingerprint=bank.fingerprint,child_fingerprint=child.fingerprint,
        diagnostic_vs_original_value_target_max_abs_error=target_parity,
        macro_unchanged=macro_before==state_digest(models['sdf_fc1'].state_dict()),targets_finite={k:bool(torch.isfinite(snapshot[k]).all()) for k in ('p0_target','pi_target','q_target')},
        raw_m_quantiles=torch.quantile(raw.flatten(),torch.tensor([0.,.5,.9,.99,1.])).tolist(),
        pv_clipping_fraction=float(((raw.flatten()<e.hyperparams.pv_m_clamp_min)|(raw.flatten()>e.hyperparams.pv_m_clamp_max)).float().mean()),
        synthetic_states=len(bank.parent_states),calculated_macro_labels_created=False,off_support='UNIFORM_BOUNDARY_CASES_NOT_CLIPPED'))
    return frame


def boundary_figures(frames,out,scatter):
    for case in sorted(frames['BEFORE'].macro_case.unique()):
        fig,axes=plt.subplots(2,2,figsize=(10,9),sharex=True,sharey=True)
        for i,eta in enumerate((0,1)):
            for j,name in enumerate(('BEFORE','AFTER_UNIFORM')):
                f=frames[name];g=f[(f.macro_case==case)&(f.eta==eta)];v=g.pivot(index='z',columns='b',values='Phat')
                ax=axes[i,j];ax.pcolormesh(v.columns,v.index,(v.to_numpy()<0),vmin=0,vmax=1,cmap='coolwarm',shading='auto')
                if v.to_numpy().min()<0<v.to_numpy().max():ax.contour(v.columns,v.index,v.to_numpy(),levels=[0],colors='black')
                if scatter is not None:ax.scatter(scatter[0],scatter[1],s=3,c='gold',alpha=.35)
                ax.set(title=f'{name}; eta={eta}; h={g.h_f.iloc[0]:.3g}, ell={g.ell_f.iloc[0]:.3g}',xlabel='b',ylabel='z')
        fig.suptitle('Black=current checkpoint boundary; identical real initial scatter; no previous-period boundary')
        fig.tight_layout();fig.savefig(out/f'four_way_boundary_macro{case}.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,eta in zip(axes,(0,1)):
        for name,color in [('BEFORE','black'),('AFTER_UNIFORM','red')]:
            g=frames[name];g=g[(g.macro_case==0)&(g.eta==eta)];v=g.pivot(index='z',columns='b',values='Phat')
            if v.to_numpy().min()<0<v.to_numpy().max():ax.contour(v.columns,v.index,v.to_numpy(),levels=[0],colors=color)
            ax.plot([],[],color=color,label=name)
        ax.set(title=f'Both checkpoints; eta={eta}',xlabel='b',ylabel='z');ax.legend()
    fig.tight_layout();fig.savefig(out/'before_after_boundary_overlay.png',dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--schedule',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--macro-support-csv',type=Path);p.add_argument('--macro-support-manifest',type=Path)
    p.add_argument('--abnormal-store',type=Path)
    p.add_argument('--parents',type=int,default=4);a=p.parse_args()
    if not 1<=a.parents<=32:p.error('bounded frozen Parents only')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);started=time.monotonic()
    def no(*args,**kwargs):raise RuntimeError('optimizer/backward forbidden in frozen comparison')
    torch.Tensor.backward=no
    for opt in (torch.optim.Adam,torch.optim.AdamW,torch.optim.LBFGS):opt.step=no
    if not a.after.exists():atomic_write_json(a.output/'summary.json',dict(status='BLOCKED_MISSING_UNIFORM_CHECKPOINT',training_performed=False));return
    before=safe_load(a.before);after=safe_load(a.after);ha,hb=matched_payloads(before,after)
    if file_hash(a.before)!='9ad7610a1851d06d80c8b2d08052a01532923f7a2808e3a006f5538b404aa94d':raise ValueError('unreviewed Before checkpoint')
    if file_hash(a.after)!='d26b9004210eef0be224c79d1f1faef6e59692e48ceba45050e616865ef2d325':raise ValueError('unreviewed Uniform checkpoint')
    bank=safe_load(a.bank);schedule=safe_load(a.schedule)
    bank=dict(bank,state=select_state(bank['state'],slice(0,a.parents),'cpu'));bank['fingerprint']=state_hash(bank['state'])
    schedule=dict(schedule,parents=schedule['parents'][:a.parents]);schedule['fingerprint']=stable_json_hash(schedule['parents']);bank['schedule_fingerprint']=schedule['fingerprint']
    extra=[];examples=[]
    if a.abnormal_store is not None:
        from data.finite_direct_long_horizon_runtime import ShardStore
        selected={}
        for f in ShardStore(a.abnormal_store).rows():
            if not {'K_cal_t1','Hatc_cal_t1','LnK_cal_t1','x_t1'}<=set(f):continue
            f=f[f.economic_state_valid & f.t.isin([1,2,3,4])]
            for t,g in f.groupby('t'):
                r=g.sort_values(['K_cal_t1','parent_candidate_id','child_branch']).iloc[0].to_dict()
                if int(t) not in selected or r['K_cal_t1']<selected[int(t)]['K_cal_t1']:selected[int(t)]=r
        for t in sorted(selected):
            r=selected[t];extra.append((r['x_t1'],r['Hatc_cal_t1'],r['LnK_cal_t1']));examples.append(r)
    atomic_write_json(a.output/'abnormal_macro_examples.json',dict(source_store_manifest_sha256=file_hash(a.abnormal_store/'manifest.json') if a.abnormal_store else None,examples=examples))
    initial=state_hash(bank['state']);grid=static_grid(extra);frames={};summaries={}
    scatter=(bank['state']['b'][0].numpy(),bank['state']['z'][0].numpy())
    # Models must retain parameter version counters for production Entry's
    # immutable-weight guard. no_grad disables graphs without inference params.
    with torch.no_grad():
        for name,cp in [('BEFORE',before),('AFTER_UNIFORM',after)]:
            out=a.output/name;out.mkdir();models,e,_,guard=setup(SimpleNamespace(device='cpu'),initial_models=cp['transaction']['models']);e._fc1_firms=200;e.refresh_firm_target()
            for m in models.values():m.eval();m.requires_grad_(False)
            e._build_simulation_bp_grid_policy=lambda:GridBPSimulationPolicy(target_model=e._target_policy_value(),sdf_fc1_model=models['sdf_fc1'],p0_loss=e.loss_fns['p0'],pi_loss=e.loss_fns['pi'],hyperparams=e.hyperparams,economic_config=e.config,n_child_shocks=e.hyperparams.simulation_bp_grid_n_child_shocks,shock_seed=getattr(e.hyperparams,'simulation_bp_grid_shock_seed',314159),require_cuda=False)
            original=hashes(models);frames[name]=evaluate_static(models,e,grid,out)
            try:
                store=generate_long_horizon_dataset(models,e,bank,schedule,{'fingerprint':'ROUND2_MATCHED_FROZEN_V1'},out/'h50',outer=12,
                    stage='POST_SDF_TRAIN_EMPIRICAL_SOURCE',parent_batch_size=4,predecision_templates=True,
                    diagnostic_metadata=dict(model_version=name,checkpoint_sha256=file_hash(a.before if name=='BEFORE' else a.after),frozen_pool_contract_only=True))
                survival=write_dashboard(store,out/'diagnostics',JointCoverageConfig())
                source=dict(path=str(store.root),outer=12,manifest_sha256=file_hash(store.root/'manifest.json'))
                templates=list(committed_templates([source],outer=13));eligible,reasons=eligible_templates(templates,outer=13,cfg=JointCoverageConfig(),committed=True)
                atomic_write_json(out/'empirical_roundtrip.json',dict(status='FROZEN_POOL_CONTRACT_NOT_ACTUAL_TRAINED_OUTER',raw=len(templates),eligible=len(eligible),unique_lineages=len({t['cluster_id'] for t in eligible}),rejections=reasons))
                summaries[name]=dict(status='COMPLETE',initial_micro_hash=initial,root_file=str(store.root/'initial_hstar_audit.csv'),
                    survival={str(t):int(survival.loc[survival.t==t,'main_spine_alive_paths'].iloc[0]) for t in (1,5,10,20,30,50)})
            except Exception as exc:
                from data.finite_direct_initial_hstar import InitialCalibrationBlocked
                if not isinstance(exc,InitialCalibrationBlocked):raise
                summaries[name]=dict(status='STRICT_INITIAL_ROOT_BLOCKED',error=str(exc),audit=exc.audit)
            if original!=hashes(models) or state_hash(bank['state'])!=initial or guard['calls']:raise ValueError('frozen invariant violation')
        boundary_figures(frames,a.output,scatter)
        if a.macro_support_csv is not None:
            if a.macro_support_manifest is None:raise ValueError('support CSV requires content-bound manifest')
            m=json.loads(a.macro_support_manifest.read_text())
            if file_hash(a.macro_support_csv)!=m['observations_sha256']:raise ValueError('support data SHA mismatch')
            from data.finite_direct_macro_contract import select_training_rows
            training=select_training_rows(pd.read_csv(a.macro_support_csv)).fc1
            support=training[['Hatc_cal_t','LnK_cal_t']].to_numpy();lo=support.min(0);hi=support.max(0)
            states=grid.parent_states[:,5:7].numpy();distance=np.maximum(lo-states,0)+np.maximum(states-hi,0)
            pd.DataFrame(dict(state_index=np.arange(len(states)),h_f=states[:,0],ell_f=states[:,1],
                h_distance_outside_D2_envelope=distance[:,0],ell_distance_outside_D2_envelope=distance[:,1],
                outside_observed_envelope=(distance>0).any(1))).to_csv(a.output/'macro_support_distance.csv',index=False)
            atomic_write_json(a.output/'macro_support_provenance.json',dict(csv_sha256=file_hash(a.macro_support_csv),manifest_sha256=file_hash(a.macro_support_manifest),
                legal_rows=len(training),h_ell_min=lo.tolist(),h_ell_max=hi.tolist(),
                interpretation='DISTANCE_TO_OBSERVED_AXIS_ENVELOPE_NOT_PROOF_OF_JOINT_SUPPORT_OR_SDF_RELIABILITY'))
    atomic_write_json(a.output/'summary.json',dict(status='COMPLETE',cases=summaries,source_hashes=dict(BEFORE=ha,AFTER_UNIFORM=hb),
        source_sha256=dict(before=file_hash(a.before),after=file_hash(a.after)),matched_macro=True,matched_input_grid=grid.fingerprint,
        independent_strict_root_per_model=True,training_performed=False,optimizer_steps=0,fc2_calls=0,
        total_seconds=time.monotonic()-started,peak_cpu_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**30 if platform.system()=='Darwin' else 2**20),
        gpu_performance='NOT_TESTED',economic_convergence='NOT_TESTED',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    print(json.dumps(summaries,default=str))


if __name__=='__main__':main()
