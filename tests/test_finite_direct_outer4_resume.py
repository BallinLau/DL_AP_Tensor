import pytest
import torch
from analysis.finite_direct_outer4_state import StageJournal
from test_finite_direct_outer4_runner import FakeBackend
from analysis.run_full_nested_finite_direct_outer4 import fresh_state,restore_fresh
from training.sdf_shock_bank import SDFShockBank
from types import SimpleNamespace


@pytest.mark.parametrize('stage',['FC1_epoch_2','SDF_chunk_1','P_block_2'])
def test_interrupted_block_restores_models_optimizer_rng_al_and_dataset(tmp_path,stage):
    b=FakeBackend(); j=StageJournal(tmp_path,{'source_commit':'fixed','data':'fingerprint'},b)
    j.run('committed_block',lambda:b.execute(0,'FC1',{}))
    expected=b.snapshot()
    def fail():
        b.state.update(value=999,optimizer=999,AL=999.,dataset='BAD')
        torch.rand(100)
        raise RuntimeError('simulated interruption')
    with pytest.raises(RuntimeError): j.run(stage,fail)
    new=FakeBackend(); resumed=StageJournal(tmp_path,j.contract,new,resume=True)
    assert new.state==expected['state']
    assert torch.equal(torch.get_rng_state(),expected['rng'])
    assert resumed.completed==['committed_block']
    called=[]; resumed.run('committed_block',lambda:called.append('duplicate'))
    assert not called
    resumed.run(stage,lambda:new.execute(0,'P',{}))
    assert new.state['optimizer']==expected['state']['optimizer']+1


def test_resume_rejects_source_or_data_mismatch(tmp_path):
    StageJournal(tmp_path,{'code':'A'},FakeBackend())
    with pytest.raises(RuntimeError,match='contract mismatch'):
        StageJournal(tmp_path,{'code':'B'},FakeBackend(),resume=True)


def test_sdf_fresh_bank_and_pair_rng_roundtrip():
    e=SimpleNamespace(device=torch.device('cpu'),_sdf_shock_bank=SDFShockBank.create(4,16,torch.device('cpu'),123),
        _sdf_pair_generator=torch.Generator().manual_seed(99),_current_epoch_idx=7)
    state=fresh_state(e)
    first=e._sdf_shock_bank.sample_pair(torch.arange(4),e._sdf_pair_generator)
    e._sdf_shock_bank.eps.zero_(); restore_fresh(e,state)
    repeated=e._sdf_shock_bank.sample_pair(torch.arange(4),e._sdf_pair_generator)
    for a,b in zip(first,repeated): torch.testing.assert_close(a,b,rtol=0,atol=0)
