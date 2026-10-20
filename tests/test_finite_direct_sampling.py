"""CPU-only sampling tests. No model forward or optimizer is executed."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
import torch

from analysis.nested_equilibrium_utils import stable_json_hash
from data.finite_direct_sampling import (
    SamplingConfig, build_or_load_sampling, capital_groups, construct_parent_bank,
    discrete_inverse_cdf, fixed_schedule, sample_b, state_hash,
    template_fingerprint, validate_template_splits,
)


@pytest.fixture
def bank():
    templates=[]
    for split,offset in [('train',0),('validation',1000)]:
        for j in range(8):
            groups=[(.02+q*.2+np.arange(10)*.001+j*.0001).tolist() for q in range(4)]
            t=dict(template_id=f'{split}:{j}',split=split,source_run='fixture',
                   source_path=offset+j,source_time=0,source_kind='TEST_ONLY',
                   b_sorted=sorted(sum(groups,[])),b_by_capital_quartile=groups)
            t['distribution_fingerprint']=template_fingerprint(t); templates.append(t)
    return dict(templates=templates,
                fingerprint=stable_json_hash([t['distribution_fingerprint'] for t in templates]),
                limitation='Synthetic test fixture; not experiment evidence.')


def test_exact_parent_mix_and_parameter_support(bank):
    for split in ('train','validation'):
        schedule=fixed_schedule(SamplingConfig(),bank,split)
        rows=schedule['parents']; n=len(rows)
        assert sum(r['sampling_source']=='BETA' for r in rows)==n//2
        for r in rows:
            if r['sampling_source']=='BETA':
                assert 1.8<=r['beta_alpha']<=3.2 and 4.8<=r['beta_beta']<=7.8
                assert r['empirical_template_id'] is None
            else: assert r['beta_alpha'] is None and r['beta_beta'] is None


def test_discrete_cdf_exact_no_interpolation():
    np.testing.assert_array_equal(discrete_inverse_cdf([.1,.4,.9],[0,.32,.34,.999]),[.1,.1,.4,.9])
    with pytest.raises(ValueError): discrete_inverse_cdf([.1],[1.])


def test_capital_conditional_cdf(bank):
    c=SamplingConfig(train_parents=8,validation_parents=4,firms=40)
    row=next(r for r in fixed_schedule(c,bank,'train')['parents'] if r['sampling_source']=='EMPIRICAL')
    template=next(t for t in bank['templates'] if t['template_id']==row['empirical_template_id'])
    capital=np.arange(1,41); b,fallback=sample_b(row,capital,template,c)
    assert fallback==0
    for q in range(4): assert np.isin(b[capital_groups(capital)==q],template['b_by_capital_quartile'][q]).all()
    b2,fallback=sample_b(row,capital,template,replace(c,conditional_min_count=11))
    assert fallback==40 and np.isin(b2,template['b_sorted']).all()


def test_split_leakage_rejected(bank):
    t=bank['templates'][-1]; t['source_path']=0
    t['distribution_fingerprint']=template_fingerprint(t)
    with pytest.raises(RuntimeError,match='leakage'): validate_template_splits(bank)


def test_source_tampering_rejected(bank):
    bank['templates'][0]['b_sorted'][0]=.7
    with pytest.raises(RuntimeError,match='corrupt'): validate_template_splits(bank)


def test_crn_and_parent_identity(bank):
    c=SamplingConfig(train_parents=8,validation_parents=4,firms=40)
    a=fixed_schedule(c,bank,'train'); b=fixed_schedule(c,bank,'validation')
    assert not ({r['parent_id'] for r in a['parents']}&{r['parent_id'] for r in b['parents']})
    for row in a['parents']:
        keys=row['child_shock_keys']
        assert [k['branch_id'] for k in keys]==[0,1]
        assert all(k['economic_time']==1 and k['path_id']==row['parent_id'] for k in keys)
    assert a==fixed_schedule(c,bank,'train')


def test_construction_reproducible_fresh_capital_and_no_optimizer(bank,monkeypatch):
    def forbidden(*a,**k): raise AssertionError('optimizer forbidden during review')
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
    monkeypatch.setattr(torch.optim.AdamW,'step',forbidden)
    c=SamplingConfig(train_parents=8,validation_parents=4,firms=40)
    s=fixed_schedule(c,bank,'train'); rng=torch.get_rng_state().clone()
    a,manifest=construct_parent_bank(c,bank,s); b,_=construct_parent_bank(c,bank,s)
    assert torch.equal(rng,torch.get_rng_state())
    assert state_hash(a)==state_hash(b)
    assert torch.isfinite(a['b']).all() and (a['b']>=0).all() and (a['b']<=1).all()
    assert not manifest.distribution_fingerprint.duplicated().any()
    assert (a['K'].sum(1)>=45-1e-4).all() and (a['K'].sum(1)<=85+1e-4).all()
    assert set(a['transition_path_id'].tolist()).isdisjoint({t['source_path'] for t in bank['templates']})
    for j,row in enumerate(s['parents']):
        if row['sampling_source']=='EMPIRICAL':
            t=next(t for t in bank['templates'] if t['template_id']==row['empirical_template_id'])
            assert np.isin(a['b'][j].numpy(),np.asarray(t['b_sorted'],dtype=np.float32)).all()
            assert not np.array_equal(a['b'][j].numpy(),np.asarray(t['b_sorted'],dtype=np.float32))


def test_resume_preserves_bank_and_schedule(bank,tmp_path):
    c=SamplingConfig(train_parents=8,validation_parents=4,firms=40)
    a,summary=build_or_load_sampling(tmp_path,c,bank,pd.DataFrame())
    b,second=build_or_load_sampling(tmp_path,c,bank,pd.DataFrame())
    assert summary==second and all(a[s]['fingerprint']==b[s]['fingerprint'] for s in a)
    with pytest.raises(RuntimeError,match='provenance'):
        build_or_load_sampling(tmp_path,replace(c,seed=123),bank,pd.DataFrame())
    path=tmp_path/'train_parent_sampling_schedule.json'; obj=json.loads(path.read_text())
    obj['parents'][0]['parent_id']+=1; path.write_text(json.dumps(obj))
    with pytest.raises(RuntimeError,match='artifact corruption'):
        build_or_load_sampling(tmp_path,c,bank,pd.DataFrame())


@pytest.mark.parametrize('change',[{'empirical_jitter':.01},{'train_parents':7},
    {'alpha_min':1.},{'conditional_min_count':0},{'capital_max':90.}])
def test_unauthorized_config_rejected(change):
    with pytest.raises(ValueError): replace(SamplingConfig(),**change).validate()
