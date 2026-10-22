"""Round3 CPU contracts. No optimizer, no formal dataset generation."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
from analysis.dstage_benchmark_support import VARIANTS,subset_inputs,BoundedStageProfiler


def test_single_factor_variants():
    baseline=asdict(VARIANTS['BASELINE_PRODUCTION']);baseline['source']='current'
    expected={'NO_TIMING_SYNC_ONLY':'timing_sync','LIGHTWEIGHT_ROOT_ONLY':'light',
              'SHARD_DIGEST_REUSE_ONLY':'digest_reuse','BP_GRID_LOCAL_ONLY':'deferred_bp_stats',
              'BATCH64_ONLY':'batch','BATCH128_ONLY':'batch'}
    for name,key in expected.items():
        differences=[k for k,v in asdict(VARIANTS[name]).items() if v!=baseline[k]]
        assert differences==[key]


def test_subset_rebinds_schedule_and_preserves_parent_order():
    from analysis.nested_equilibrium_utils import stable_json_hash
    from data.finite_direct_sampling import state_hash
    bank={'state':{'x':torch.arange(4.),'transition_path_id':torch.tensor([17,3,45,9])},
          'fingerprint':'old','schedule_fingerprint':'old'}
    schedule={'seed':123,'parents':[{'parent_id':i} for i in [45,9,3,17]],'fingerprint':'old'}
    original=deepcopy(schedule)
    b,s=subset_inputs(bank,schedule,2)
    assert [r['parent_id'] for r in s['parents']]==[17,3]
    assert s['fingerprint']==stable_json_hash(s['parents'])==b['schedule_fingerprint']
    assert b['fingerprint']==state_hash(b['state'])
    assert s['seed']==123 and schedule==original
    assert bank['fingerprint']=='old'


@pytest.mark.parametrize('parents',[[{'parent_id':1},{'parent_id':1}],[{'parent_id':3}]])
def test_bad_mapping_rejected(parents):
    bank={'state':{'x':torch.zeros(2),'transition_path_id':torch.tensor([1,2])}}
    with pytest.raises(ValueError):subset_inputs(bank,{'parents':parents},2)


def test_plan_only_does_not_load_models_or_artifacts(tmp_path):
    root=Path(__file__).resolve().parents[1]
    cmd=[sys.executable,str(root/'analysis/benchmark_dstage_performance.py')]
    for name in ['checkpoint','bank','schedule','initial-root-cache','source-original','source-round2']:
        cmd+=['--'+name,str(tmp_path/'MISSING')]
    cmd+=['--output',str(tmp_path/'plan')]
    result=subprocess.run(cmd,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    plan=json.loads((tmp_path/'plan/benchmark_plan.json').read_text())
    assert plan['independent_processes'] and plan['formal_training'] is False
    assert plan['combined']=='BLOCKED_PENDING_FULL_PATH_PARITY'
    assert 'COMBINED_OPTIMIZATION' not in plan['variants']


def test_profile_is_bounded_and_restores_patch():
    owner=SimpleNamespace(f=lambda tensor:tensor+1)
    original=owner.f
    profiler=BoundedStageProfiler('cpu',max_regions=1)
    profiler.patch(owner,'f','CPU_TEST')
    assert torch.equal(owner.f(torch.zeros(2)),torch.ones(2))
    with pytest.raises(RuntimeError,match='region limit'):owner.f(torch.zeros(2))
    profiler.restore();assert owner.f is original
    report=profiler.finish()
    assert report['nested_regions_are_additive'] is False
    assert report['regions'][0]['cuda_stream_span_seconds'] is None


def test_deferred_grid_statistics_preserve_actions_rng_and_counts(monkeypatch):
    from test_bp_head_vs_grid_simulation import _GridTeacher,_GridTarget,_GridSDF
    from training.bp_simulation_policy import GridBPSimulationPolicy
    from config import Config
    monkeypatch.setattr('training.bp_simulation_policy.BPGridTeacher.from_hyperparams',
                        lambda *args,**kwargs:_GridTeacher())
    model,sdf=_GridTarget(),_GridSDF()
    kwargs=dict(target_model=model,sdf_fc1_model=sdf,p0_loss=object(),pi_loss=object(),
        hyperparams=SimpleNamespace(pv_use_clipped_m=False),economic_config=Config,
        n_child_shocks=2,shock_seed=77,require_cuda=False)
    normal=GridBPSimulationPolicy(**kwargs)
    deferred=GridBPSimulationPolicy(**kwargs,defer_diagnostic_sync=True)
    state=torch.tensor([[.2,-.1,1.,.1,0.,-2.,4.],[.4,.1,0.,.2,0.,-1.,3.]])
    rng=torch.random.get_rng_state().clone()
    for _ in range(3):
        a=normal.evaluate_actions(firm_state=state,hatc_cal=state[:,5:6],lnk_cal=state[:,6:7])
        b=deferred.evaluate_actions(firm_state=state,hatc_cal=state[:,5:6],lnk_cal=state[:,6:7])
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    # Actual host counters remain untouched until explicit collection.
    assert deferred.refinancing_active_rows==0
    expected=normal.instrumentation()
    assert deferred.instrumentation()==expected
    assert deferred.instrumentation()==expected  # no double counting on reread
    deferred.evaluate_actions(firm_state=state,hatc_cal=state[:,5:6],lnk_cal=state[:,6:7])
    assert deferred.instrumentation()['refinancing_active_rows']==4
    assert torch.equal(torch.random.get_rng_state(),rng)
    normal.verify_immutable();deferred.verify_immutable()
