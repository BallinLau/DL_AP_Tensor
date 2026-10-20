"""Finite-direct macro row eligibility; no imputation or economics here.

FC1 can consume one valid child. Signed AiO consumes complete parent/two-child
pairs only. Invalid observations remain in the returned audit table.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch


@dataclass
class MacroRows:
    fc1: pd.DataFrame
    sdf_pairs: pd.DataFrame
    audit: pd.DataFrame


def select_training_rows(frame):
    keys=['split','parent_candidate_id','t','child_branch']
    required=keys+['economic_state_valid','x_t','x_t1','Hatc_cal_t','LnK_cal_t',
                   'Hatc_cal_t1','LnK_cal_t1','C_raw_t1','K_cal_t1']
    missing=set(required)-set(frame.columns)
    if missing: raise ValueError(f'missing finite-direct columns: {sorted(missing)}')
    if frame.duplicated(keys).any(): raise ValueError('duplicate economic child')
    if not frame.child_branch.isin([0,1]).all(): raise ValueError('child branch must be 0/1')
    ids=frame[['parent_candidate_id','t']].to_numpy(dtype=float)
    if not np.isfinite(ids).all() or not np.equal(ids,np.floor(ids)).all() or (ids[:,0]<0).any() or (ids[:,1]<1).any():
        raise ValueError('undefined economic parent/time identity')
    if not frame.split.isin(['train','validation']).all(): raise ValueError('unknown split')
    train=set(frame.loc[frame.split=='train','parent_candidate_id'])
    val=set(frame.loc[frame.split=='validation','parent_candidate_id'])
    if train&val: raise ValueError('train/validation parent lineage leakage')
    if not frame.economic_state_valid.isin([True,False]).all():
        raise ValueError('validity flag must be boolean, not missing/string')
    numeric=required[5:]
    finite=np.isfinite(frame[numeric].to_numpy(dtype=float)).all(axis=1)
    legal=(frame.C_raw_t1>0)&(frame.K_cal_t1>0)&finite
    claimed=frame.economic_state_valid.astype(bool)
    if (claimed&~legal).any(): raise ValueError('valid child has undefined economic label')
    valid=frame.loc[claimed]
    if not np.allclose(valid.Hatc_cal_t1,np.log(valid.C_raw_t1/valid.K_cal_t1),rtol=0,atol=2e-6):
        raise ValueError('calculated Hatc disagrees with raw resource ledger')
    if not np.allclose(valid.LnK_cal_t1,np.log(valid.K_cal_t1),rtol=0,atol=2e-6):
        raise ValueError('calculated LnK disagrees with capital ledger')
    audit=frame.copy(); audit['fc1_eligible']=claimed&legal
    group=['split','parent_candidate_id','t']
    # Parent macro context must be identical across the two siblings, including
    # rows that failed later. Never average two inconsistent parent states.
    for _,siblings in audit.groupby(group):
        valid=siblings.loc[siblings.fc1_eligible]
        if len(valid)>1:
            parent=valid[['x_t','Hatc_cal_t','LnK_cal_t']].to_numpy()
            if not np.array_equal(parent,parent[:1].repeat(len(parent),axis=0)):
                raise ValueError('siblings disagree on true parent context')
    counts=audit.groupby(group).fc1_eligible.transform('sum')
    audit['sdf_eligible']=audit.fc1_eligible&(counts==2)
    return MacroRows(audit.loc[audit.fc1_eligible].copy(),
                     audit.loc[audit.sdf_eligible].copy(),audit)


def production_sdf_tensors(frame,device='cpu'):
    """Production parent[9]/children[10], with TRUE parent C/K in cols7/8.

    Firm-specific first seven columns are placeholders for this macro-only
    adapter; production SDF reads x and true macro columns, never Firm rows.
    Returns contiguous parent_index IDs separately from global economic IDs.
    """
    rows=select_training_rows(frame).sdf_pairs
    parents=[]; children=[[],[]]; lineage=[]
    for (split,pid,t),g in rows.groupby(['split','parent_candidate_id','t'],sort=True):
        g=g.sort_values('child_branch'); a,b=g.iloc[0],g.iloc[1]
        p=[0.]*9; p[4]=a.x_t; p[7]=a.Hatc_cal_t; p[8]=a.LnK_cal_t
        # No fabricated forecast in a TRUE_ONLY tensor: NaN makes accidental
        # routing to bootstrap/recursive input fail instead of training on zero.
        p[5]=a.get('Hatcf_t',float('nan')); p[6]=a.get('LnKF_t',float('nan'))
        parents.append(p); lineage.append((split,int(pid),int(t)))
        for branch,row in enumerate((a,b)):
            c=[0.]*10; c[4]=row.x_t1
            c[5]=row.get('Hatcf_t1',float('nan')); c[6]=row.get('LnKF_t1',float('nan'))
            c[7]=1.; c[8]=row.Hatc_cal_t1; c[9]=row.LnK_cal_t1
            children[branch].append(c)
    if not parents: raise ValueError('no complete parent/two-child macro pairs')
    p=torch.tensor(parents,dtype=torch.float32,device=device)
    c=[torch.tensor(x,dtype=torch.float32,device=device) for x in children]
    return dict(parents=p,children=c,lineage=lineage,
                parent_index=torch.arange(len(p),device=device))
