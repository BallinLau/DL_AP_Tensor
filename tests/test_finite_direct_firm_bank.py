import pytest
import torch
from data.finite_direct_runtime import firm_bank_from_captures,check_firm_lineage,conditional_child_bank
from analysis.economic_config import AnalysisEconomicConfig


def snapshot(split,path):
    n=4; states=torch.zeros(n,7); states[:,0]=torch.arange(n)/10
    states[:,4]=.12; states[:,5]=-2.; states[:,6]=4.
    return dict(split=split,source_time=0,states=states,context=torch.tensor([[-1.,3.]]).repeat(n,1),
        eligible=torch.ones(n,dtype=torch.bool),path=torch.full((n,),path),firm_id=torch.arange(n),model_hashes={'test':'source'})


def test_observed_rows_not_synthetic_marginal_mix():
    s=snapshot('train',1); b=firm_bank_from_captures([s],split='train',max_rows=4,seed=3)
    assert {tuple(x) for x in b.parent_states.tolist()}=={tuple(x) for x in s['states'].tolist()}
    val=firm_bank_from_captures([snapshot('validation',2)],split='validation',max_rows=4,seed=3)
    check_firm_lineage(b,val)
    val.metadata['path'].fill_(1)
    with pytest.raises(RuntimeError,match='leakage'): check_firm_lineage(b,val)


def test_conditional_transition_uses_same_node_true_macro_not_forecast():
    class Macro(torch.nn.Module):
        def forward_step(self,*,x_prev,x_curr,hatcf_prev,lnkf_prev,return_physical):
            assert (hatcf_prev==-1.).all() and (lnkf_prev==3.).all()
            shape=x_curr.shape; ones=torch.ones(shape)
            return ones[:,0],ones,ones*.9,hatcf_prev[:,None].expand(shape),lnkf_prev[:,None].expand(shape)
    bank=firm_bank_from_captures([snapshot('train',1)],split='train',max_rows=4,seed=3)
    c=conditional_child_bank(Macro(),bank,seed=5,n_child_shocks=2,
        economic=AnalysisEconomicConfig.from_current_config(),device='cpu',chunk_size=2)
    assert c.children.shape==(4,2,8)
    assert (c.children[:,:,5]==-1).all() and (c.children[:,:,6]==3).all()
    assert (c.children[:,:,7]==.9).all() and c.metadata['eta']=='integrated_exactly_in_operator'
