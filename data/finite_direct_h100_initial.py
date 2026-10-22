"""Bounded stratified exploration designs; never counterfeit cleared labels."""
from dataclasses import replace
import math
import torch
from config import Config
from data.finite_direct_joint_coverage import JointCoverageConfig,propose_beta_batch
from data.finite_direct_sampling import state_hash
from analysis.nested_equilibrium_utils import stable_json_hash


def design_initial(reference,*,outer,mode,seed=20261022,count=100):
    if count!=100:raise ValueError('production H100 design requires 100+100 paths')
    if mode not in ('ROOT_CONSISTENT','OFF_EQUILIBRIUM_INITIAL_BELIEF'):raise ValueError('invalid initial mode')
    offset=0 if mode=='ROOT_CONSISTENT' else 100
    cfg=replace(JointCoverageConfig(),seed=seed)
    points=torch.quasirandom.SobolEngine(5,scramble=True,seed=seed+outer+offset).draw(count).double()
    s,specs=propose_beta_batch(reference,start=offset,count=count,outer=outer,cfg=cfg,design_points=points)
    if s['K'].shape[1]!=200:raise ValueError('H100 requires 200 firms')
    # No changed stationary marginal. Sobol coordinates select stationary x
    # quantiles, original seven b families/rank modes, and real capital scale.
    q=points[:,2].clamp(1e-9,1-1e-9)
    standard=torch.distributions.Normal(0.,1.).icdf(q)
    s['x']=(float(Config.XBAR)+float(Config.SIGMA_X)/math.sqrt(1-float(Config.RHO_X)**2)*standard).float()
    target_ell=-2+7*points[:,1]
    weights=s['K'].double()/s['K'].double().sum(1,keepdim=True)
    s['K']=(weights*target_ell.exp()[:,None]).float();s['K_birth']=s['K'].clone()
    s['lnkf']=s['K'].sum(1).log()
    if mode=='OFF_EQUILIBRIUM_INITIAL_BELIEF':s['hatcf']=(-6+8*points[:,0]).float()
    actual_seed=seed+outer*100003+offset
    for j,p in enumerate(specs):
        p.update(initial_mode=mode,K_total_requested=float(target_ell[j].exp()),
            K_total_realized=float(s['K'][j].sum()),sobol=points[j].tolist())
        for shock in p['child_shock_keys']:
            shock.update(transition_seed=actual_seed+11,entry_seed=actual_seed+29)
    schedule=dict(split='train',seed=actual_seed,parents=specs)
    schedule['fingerprint']=stable_json_hash(schedule)
    bank=dict(state=s,fingerprint=state_hash(s),schedule_fingerprint=schedule['fingerprint'])
    metadata=dict(mode=mode,count=count,seed=seed,outer=outer,design='SCRAMBLED_SOBOL',
        state_fingerprint=bank['fingerprint'],schedule_fingerprint=schedule['fingerprint'],
        initial_ids=s['transition_path_id'].tolist(),root_required=mode=='ROOT_CONSISTENT',
        initial_belief_is_not_label=True)
    return bank,schedule,metadata
