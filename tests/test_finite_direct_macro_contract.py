import pandas as pd
import math
import pytest
from data.finite_direct_macro_contract import select_training_rows,production_sdf_tensors


def rows():
    out=[]
    for pid in range(3):
        for branch in (0,1):
            out.append(dict(split='train',parent_candidate_id=pid,t=1,child_branch=branch,
                economic_state_valid=True,x_t=.1,x_t1=.2+branch*.03,
                Hatc_cal_t=-1.,LnK_cal_t=4.,Hatc_cal_t1=math.log(20/60),LnK_cal_t1=math.log(60),
                C_raw_t1=20.,K_cal_t1=60.,Hatcf_t1=-1.2,LnKF_t1=4.2))
    return pd.DataFrame(out)


def test_single_child_for_fc1_not_sdf():
    f=rows(); f.loc[1,'economic_state_valid']=False; f.loc[1,'C_raw_t1']=-1
    selected=select_training_rows(f)
    assert len(selected.fc1)==5 and len(selected.sdf_pairs)==4 and len(selected.audit)==6
    tensors=production_sdf_tensors(f)
    assert tensors['parents'].shape==(2,9)
    assert all(x.shape==(2,10) for x in tensors['children'])
    assert tensors['parents'][:,7].tolist()==[-1.,-1.]
    assert tensors['parent_index'].tolist()==[0,1]


def test_no_valid_pairs_is_hard_error():
    f=rows(); f.loc[f.child_branch==1,'economic_state_valid']=False
    with pytest.raises(ValueError,match='no complete'): production_sdf_tensors(f)


@pytest.mark.parametrize('fault',['duplicate','lineage','parent','nonfinite','string','time','ledger'])
def test_corruption_rejected(fault):
    f=rows()
    if fault=='duplicate': f=pd.concat([f,f.iloc[:1]])
    if fault=='lineage': f.loc[1,'split']='validation'
    if fault=='parent': f.loc[1,'Hatc_cal_t']=-2.
    if fault=='nonfinite': f.loc[1,'Hatc_cal_t1']=float('nan')
    if fault=='string': f['economic_state_valid']='False'
    if fault=='time': f.loc[1,'t']=float('nan')
    if fault=='ledger': f.loc[1,'Hatc_cal_t1']+=.01
    with pytest.raises(ValueError): select_training_rows(f)
