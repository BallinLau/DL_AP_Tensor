"""Wide Firm collocation, NOT a simulated aggregate population distribution."""
from copy import deepcopy
from dataclasses import dataclass,asdict
from collections import Counter
import torch
from training.inner_fixed_point import InnerFPCollocationBank


def sampling_spec():
    return dict(version='FIRM_UNIFORM_MACRO_V1',distribution='INDEPENDENT_IID_UNIFORM',
                hatcf_bounds=[-6.,2.],lnkf_bounds=[-2.,5.],
                scope='FIRM_COLLOCATION_ONLY_NOT_MACRO_LABELS',
                transition_context='SYNTHETIC_CURRENT_MACRO_PERCEIVED_FC1_TRANSITION',
                micro_columns_preserved=[0,1,2,3,4],default_firm_batch_size=2048)


def uniform_macro_bank(bank, *, seed, hatcf_bounds=(-6.,2.), lnkf_bounds=(-2.,5.)):
    """Preserve joint micro donors, replace h/K, invalidate targets by fingerprint.

    Actual observed donor macro context is provenance only. Continuations must
    be rebuilt at the synthetic current macro context, never at the old donor.
    No claim that this synthetic macro/micro combination clears markets.
    """
    bounds=torch.tensor([hatcf_bounds,lnkf_bounds],dtype=bank.parent_states.dtype)
    if bounds.shape!=(2,2) or not torch.isfinite(bounds).all() or not (bounds[:,1]>bounds[:,0]).all():
        raise ValueError('finite increasing macro bounds required')
    if bank.parent_states.device.type!='cpu':
        raise ValueError('construct collocation on CPU before GPU batching')
    generator=torch.Generator(device='cpu').manual_seed(seed)
    states=bank.parent_states.detach().clone()
    values=torch.rand((len(states),2),generator=generator,dtype=states.dtype)
    states[:,5:7]=bounds[:,0]+values*(bounds[:,1]-bounds[:,0])
    meta=deepcopy(bank.metadata)
    meta['donor_parent_fingerprint']=bank.fingerprint
    meta['donor_source_kind']=meta.get('source_kind')
    meta['donor_calculated_fraction']=meta.pop('calculated_fraction',None)
    meta['donor_macro_inputs']=bank.parent_states[:,5:7].detach().clone()
    observed=meta.pop('true_macro_context',None)
    if observed is not None:meta['donor_observed_macro_context']=observed
    meta['transition_macro_context']=states[:,5:7].clone()
    meta['macro_context_kind']='SYNTHETIC_CURRENT_MACRO_PERCEIVED_FC1_TRANSITION'
    meta['source_kind']=['FIRM_UNIFORM_MACRO_COLLOCATION']*len(states)
    meta['synthetic_macro_mask']=torch.ones(len(states),dtype=torch.bool)
    meta['source_composition']={'FIRM_UNIFORM_MACRO_COLLOCATION':len(states)}
    meta['uniform_macro_sampling']=dict(sampling_spec(),seed=seed,
        hatcf_bounds=list(hatcf_bounds),lnkf_bounds=list(lnkf_bounds))
    meta['target_contract']='REBUILD_CONDITIONAL_CHILD_BANK_AND_BELLMAN_RHS; NO_CACHED_LABELS'
    meta['population_distribution']=False
    return InnerFPCollocationBank(states,torch.full_like(bank.source_kind,2),seed,bank.split,meta)


@dataclass(frozen=True)
class FirmMacroCoverageConfig:
    firm_macro_coverage_enabled:bool=False
    firm_macro_coverage_mode:str='HYBRID'
    firm_macro_h_bounds:tuple=(-6.,2.)
    firm_macro_ell_bounds:tuple=(-2.,5.)
    firm_macro_sampling_seed:int=20261022
    firm_macro_batch_size:int=2048
    firm_macro_inner_cycles:int=3
    firm_macro_coverage_fraction:float|None=None
    firm_macro_sampling_budget:int=8192
    def validate(self):
        if self.firm_macro_coverage_mode not in ('STANDALONE','HYBRID'):raise ValueError('invalid Firm coverage mode')
        if min(self.firm_macro_batch_size,self.firm_macro_inner_cycles,self.firm_macro_sampling_budget)<=0:raise ValueError('invalid Firm budget')
        if self.firm_macro_inner_cycles!=3:raise ValueError('reviewed three-inner budget required')
        if self.firm_macro_coverage_enabled:
            f=self.firm_macro_coverage_fraction
            if f is None or not 0<=f<=1:raise ValueError('explicit Uniform fraction required')
            if self.firm_macro_coverage_mode=='STANDALONE' and f!=1:raise ValueError('standalone requires fraction=1')
        for b in (self.firm_macro_h_bounds,self.firm_macro_ell_bounds):
            if len(b)!=2 or not torch.isfinite(torch.tensor(b)).all() or b[0]>=b[1]:raise ValueError('invalid macro bounds')


def bounded_donor_bank(bank,rows,*,seed):
    """Shared B/C donor selection, no invented micro states or target reuse."""
    n=len(bank.parent_states)
    if rows<1 or rows>n:raise ValueError('insufficient real donor capacity for declared sampling budget')
    if rows==n:return bank
    ids=torch.randperm(n,generator=torch.Generator().manual_seed(seed))[:rows]
    meta=deepcopy(bank.metadata)
    for key,v in list(meta.items()):
        if torch.is_tensor(v) and v.ndim and len(v)==n:meta[key]=v[ids].clone()
        elif isinstance(v,list) and len(v)==n:meta[key]=[v[j] for j in ids.tolist()]
    meta['donor_selection_indices']=ids
    meta['unique_source_firm_time_count']=micro_donor_count(meta)
    return InnerFPCollocationBank(bank.parent_states[ids].clone(),bank.source_kind[ids].clone(),seed,bank.split,meta)


def configured_macro_bank(bank,cfg,*,seed):
    """Retain residual ON_POLICY/CALCULATED 50:50; Uniform is explicit only.

    Stratification changes only which REAL donor rows receive the synthetic
    context. Every donor remains identifiable; no aggregate calculated label.
    """
    cfg.validate()
    if not cfg.firm_macro_coverage_enabled:return bank
    n=len(bank.parent_states);nuni=round(n*cfg.firm_macro_coverage_fraction)
    if cfg.firm_macro_coverage_mode=='STANDALONE':nuni=n
    g=torch.Generator().manual_seed(seed)
    # Interleave occupied eta/b/z strata; avoid one survivor stratum taking all.
    p=bank.parent_states;bc=torch.bucketize(p[:,0].contiguous(),torch.tensor([.2,.4,.6,.8]))
    zc=torch.bucketize(p[:,1].contiguous(),torch.quantile(p[:,1],torch.tensor([.25,.5,.75])))
    strata=p[:,2].long()*20+bc*4+zc
    pools=[torch.nonzero(strata==k).flatten()[torch.randperm(int((strata==k).sum()),generator=g)] for k in torch.unique(strata)]
    order=torch.cat([v for v in pools])
    # Sort by within-stratum rank first, then reproducibly randomized stratum.
    ranks=torch.cat([torch.arange(len(v)) for v in pools]);order=order[torch.argsort(ranks,stable=True)]
    leaders=[]
    for eta in (0,1):
        matches=order[p[order,2]==eta]
        if len(matches):leaders.append(matches[0])
    if nuni and nuni<len(leaders):raise ValueError('Uniform budget cannot cover both available eta strata')
    if leaders:
        leaders=torch.stack(leaders);order=torch.cat([leaders,order[~torch.isin(order,leaders)]])
    uniform_ids=order[:nuni];remaining_mask=torch.ones(n,dtype=torch.bool);remaining_mask[uniform_ids]=False
    residual=n-nuni;ncal=residual//2;non=residual-ncal
    kinds=bank.metadata['source_kind']
    calc=torch.tensor([j for j,k in enumerate(kinds) if 'CALCULATED' in k],dtype=torch.long)
    on=torch.tensor([j for j,k in enumerate(kinds) if 'CALCULATED' not in k],dtype=torch.long)
    # Select distinct residual indices; Uniform may reuse micro donor identity
    # already in its own category, disclosed in composition rather than hidden.
    if len(calc)<ncal or len(on)<non:raise ValueError('insufficient original 50:50 donor capacity')
    ids=torch.cat([on[:non],calc[:ncal],uniform_ids])
    states=p[ids].clone();meta=deepcopy(bank.metadata)
    for key,v in list(meta.items()):
        if torch.is_tensor(v) and v.ndim and len(v)==n:meta[key]=v[ids].clone()
        elif isinstance(v,list) and len(v)==n:meta[key]=[v[j] for j in ids.tolist()]
    observed=meta['true_macro_context'].clone();context=observed.clone()
    mask=torch.arange(n)>=residual
    values=torch.rand((nuni,2),generator=g,dtype=states.dtype)
    bounds=torch.tensor([cfg.firm_macro_h_bounds,cfg.firm_macro_ell_bounds],dtype=states.dtype)
    states[mask,5:7]=bounds[:,0]+values*(bounds[:,1]-bounds[:,0]);context[mask]=states[mask,5:7]
    meta.update(macro_context_kind='MIXED_FIRM_COLLOCATION_V1',transition_macro_context=context,
        synthetic_macro_mask=mask,source_kind=['FIRM_ON_POLICY']*non+['FIRM_CALCULATED_MACRO_COLLOCATION']*ncal+['FIRM_UNIFORM_MACRO_COLLOCATION']*nuni,
        firm_macro_config=asdict(cfg),target_contract='REBUILD_CONDITIONAL_CHILD_BANK_AND_BELLMAN_RHS; NO_CACHED_LABELS',
        source_composition=dict(FIRM_ON_POLICY=non,FIRM_CALCULATED_MACRO_COLLOCATION=ncal,FIRM_UNIFORM_MACRO_COLLOCATION=nuni),
        synthetic_rows_are_calculated_labels=False,population_distribution=False)
    source_codes=bank.source_kind[ids].clone();source_codes[mask]=2
    meta['unique_selected_bank_indices']=len(torch.unique(ids))
    meta['actual_unique_micro_donors']=micro_donor_count(meta)
    return InnerFPCollocationBank(states,source_codes,seed,bank.split,meta)


def firm_coverage_summary(bank):
    p=bank.parent_states
    labels=[k if k.startswith('FIRM_') else 'FIRM_'+k for k in bank.metadata.get('source_kind',[])]
    return dict(rows=len(p),sources=dict(Counter(labels)),actual_unique_micro_donors=micro_donor_count(bank.metadata),
        eta_counts={str(v):int((p[:,2]==v).sum()) for v in (0,1)},
        b_hist=torch.histc(p[:,0],bins=5,min=0,max=1).tolist(),
        z_quantiles=torch.quantile(p[:,1],torch.tensor([0.,.1,.5,.9,1.])).tolist(),
        h_hist=torch.histc(p[:,5],bins=8,min=-6,max=2).tolist(),
        ell_hist=torch.histc(p[:,6],bins=7,min=-2,max=5).tolist(),
        synthetic_proposed=int(bank.metadata.get('synthetic_macro_mask',torch.zeros(len(p))).sum()),
        finite_parent_count=int(torch.isfinite(p).all(1).sum()),rejected_states=0,
        target_finite_states='CHECK_AT_FRESH_TARGET_EVALUATION_NOT_ASSUMED',off_support='REPORTED_NOT_CLIPPED')


def micro_donor_count(meta):
    fields=('path','source_time','source_branch','firm_id')
    if not all(k in meta for k in fields):return None
    return len(torch.unique(torch.stack([meta[k].long() for k in fields],1),dim=0))
