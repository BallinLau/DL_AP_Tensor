"""Frozen 128-Parent H10 benchmark. Never executes an optimizer or formal runner."""
import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json, resource, sys, time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import pandas as pd
from analysis.finite_direct_continuation_state import safe_load
from analysis.run_fc1_self_consistent_macro import setup, select_state, hashes
from analysis.nested_equilibrium_utils import atomic_write_json
from data.finite_direct_sampling import state_hash
from data.market_clearing import MarketClearingConfig
import data.finite_direct_replay_runtime as runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','bank','schedule','output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('require a fresh independent benchmark output')
    a.output.mkdir(parents=True)
    def forbidden(*args, **kwargs):
        raise RuntimeError('BENCHMARK_OPTIMIZATION_FORBIDDEN')
    for cls in [torch.optim.Adam, torch.optim.AdamW, torch.optim.LBFGS]:
        cls.step = forbidden
    torch.Tensor.backward = forbidden
    torch.set_num_threads(8)
    cp = safe_load(a.checkpoint)
    learning = cp.get('backend', cp)
    models, episode, _, guard = setup(SimpleNamespace(device='cuda:0'),
                                     initial_models=learning['transaction']['models'])
    episode._fc1_firms = 200
    episode.refresh_firm_target()
    before = hashes(models)
    bank, schedule = safe_load(a.bank), safe_load(a.schedule)
    state = select_state(bank['state'], slice(0,128), torch.device('cpu'))
    if len(state['x']) != 128:
        raise ValueError('same 128 actual Parents required')
    bank = dict(bank, state=state, fingerprint=state_hash(state))
    ids = set(state['transition_path_id'].tolist())
    schedule = dict(schedule, parents=[r for r in schedule['parents'] if r['parent_id'] in ids])
    results = []
    reference = None
    for name,batch,light,io in [('ORIGINAL',32,False,False),
            ('OPTIMIZED_ROOT_FORWARD',32,True,False),
            ('OPTIMIZED_BATCH_64',64,True,False),
            ('OPTIMIZED_BATCH_128',128,True,False),
            ('OPTIMIZED_IO',32,False,True),
            ('COMBINED_OPTIMIZED',64,True,True)]:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        hash_stats = dict(calls=0,bytes=0,seconds=0.)
        original_hash = runtime.file_hash
        def counted_hash(path):
            t = time.monotonic(); digest = original_hash(path)
            hash_stats['calls'] += 1; hash_stats['bytes'] += Path(path).stat().st_size
            hash_stats['seconds'] += time.monotonic()-t
            return digest
        runtime.file_hash = counted_hash
        try:
            store = runtime.generate_long_horizon_dataset(models,episode,bank,schedule,
                {'fingerprint':'FROZEN_D_STAGE_PERFORMANCE_128_V1'},a.output/name,
                outer=12,stage='FROZEN_BENCHMARK',horizon=10,branches=2,capture=False,
                parent_batch_size=batch,reuse_shard_digest=io,
                root_config=MarketClearingConfig(lightweight_hard_candidates=light,
                                                profile_root_timing=name=='ORIGINAL'))
            torch.cuda.synchronize()
            seconds = time.monotonic()-started
            frame = pd.concat(list(store.rows()),ignore_index=True)
            keys = ['parent_candidate_id','t','child_branch']
            frame = frame.sort_values(keys).reset_index(drop=True)
            if reference is None:
                reference = frame
                parity = 'REFERENCE'
            else:
                # No relaxed tolerances. Detailed exception is retained on failure.
                pd.testing.assert_frame_equal(reference,frame,check_exact=True)
                parity = 'BITWISE_EQUAL'
            result = dict(variant=name,batch=batch,wall_seconds=seconds,
                parents_per_second=128/seconds,parity=parity,
                peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                cpu_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                hash_stats=hash_stats,shards=len(store.manifest['shards']),
                valid_child_rows=int(frame.economic_state_valid.sum()),
                bytes_written=sum(f.stat().st_size for f in (a.output/name).rglob('*') if f.is_file()))
        except torch.cuda.OutOfMemoryError as exc:
            result = dict(variant=name,status='OOM',error=str(exc),
                          peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
            torch.cuda.empty_cache()
        except AssertionError as exc:
            result = dict(variant=name,status='FAILED_PARITY',error=str(exc),
                          peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
            frame.to_csv(a.output/(name+'_failed_rows.csv'),index=False)
        finally:
            runtime.file_hash = original_hash
        results.append(result)
        atomic_write_json(a.output/'benchmark.json',dict(results=results,
            training_performed=False,model_hashes=before,fc2_calls=guard['calls'],
            root_cache='BLOCKED_PENDING_DEPENDENCY_AND_FULL_PATH_AB'))
        if hashes(models) != before:
            raise RuntimeError('frozen models mutated')
        print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
