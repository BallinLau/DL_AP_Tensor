"""Frozen benchmark bookkeeping, independent of production economic equations."""
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import functools
import time
import torch


@dataclass(frozen=True)
class Variant:
    source: str = 'current'
    batch: int = 32
    timing_sync: bool = True
    light: bool = False
    digest_reuse: bool = False
    deferred_bp_stats: bool = False


VARIANTS = {
    'BASELINE_PRODUCTION': Variant(source='original'),
    'ROUND2_BASELINE': Variant(source='round2', timing_sync=False, digest_reuse=True),
    'NO_TIMING_SYNC_ONLY': Variant(timing_sync=False),
    'LIGHTWEIGHT_ROOT_ONLY': Variant(light=True),
    'SHARD_DIGEST_REUSE_ONLY': Variant(digest_reuse=True),
    'BP_GRID_LOCAL_ONLY': Variant(deferred_bp_stats=True),
    'BATCH64_ONLY': Variant(batch=64),
    'BATCH128_ONLY': Variant(batch=128),
    # Deliberately not eligible until explicitly certified by full-path parity.
    'COMBINED_OPTIMIZATION': Variant(batch=32, timing_sync=False, digest_reuse=True,
                                     deferred_bp_stats=True),
}


def subset_inputs(bank, schedule, count):
    """Keep first recorded Parents and rebuild BOTH linked content identities."""
    from analysis.run_fc1_self_consistent_macro import select_state
    from analysis.nested_equilibrium_utils import stable_json_hash
    from data.finite_direct_sampling import state_hash
    state = select_state(bank['state'], slice(0, count), torch.device('cpu'))
    ids = state['transition_path_id'].tolist()
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError('insufficient or duplicate recorded Parent identities')
    by_id = {r['parent_id']: r for r in schedule['parents']}
    if len(by_id) != len(schedule['parents']) or any(i not in by_id for i in ids):
        raise ValueError('ambiguous Bank/Schedule Parent mapping')
    parents = [by_id[i] for i in ids]
    fingerprint = stable_json_hash(parents)
    return (dict(bank, state=state, fingerprint=state_hash(state),
                 schedule_fingerprint=fingerprint),
            dict(schedule, parents=parents, fingerprint=fingerprint))


class BoundedStageProfiler:
    """Nested host-call windows plus delayed CUDA-event spans, not exclusive CPU time.

    No device synchronization inside a region. CUDA spans can overlap across
    nested regions and include stream gaps; never sum them as kernel totals.
    Actual kernel attribution requires the optional torch.profiler trace.
    """
    def __init__(self, device='cpu', max_regions=20000):
        self.device = torch.device(device)
        self.max_regions = max_regions
        self.records = []
        self.patches = []
        self._entered = 0

    @contextmanager
    def region(self, name, tensors=()):
        if self._entered >= self.max_regions:
            raise RuntimeError('bounded profiler region limit reached')
        self._entered += 1
        events = None
        if self.device.type == 'cuda':
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
        started = time.perf_counter()
        with torch.profiler.record_function(name):
            try:
                yield
            finally:
                if events:
                    events[1].record()
                self.records.append(dict(stage=name,host_elapsed_seconds=time.perf_counter()-started,
                    shapes=[list(t.shape) for t in tensors if torch.is_tensor(t)],events=events))

    def patch(self, owner, name, label):
        original = getattr(owner, name)
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            tensors = tuple(x for x in args if torch.is_tensor(x))
            stage = label(*args, **kwargs) if callable(label) else label
            with self.region(stage, tensors):
                return original(*args, **kwargs)
        setattr(owner,name,wrapper)
        self.patches.append((owner,name,original))

    def finish(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        output = []
        for record in self.records:
            item = {k:v for k,v in record.items() if k!='events'}
            item['cuda_stream_span_seconds'] = (
                record['events'][0].elapsed_time(record['events'][1])/1000
                if record['events'] else None)
            output.append(item)
        return dict(regions=output, exclusive_cpu_time='NOT_MEASURED',
                    cuda_kernel_time='USE_TORCH_PROFILER_TRACE',
                    nested_regions_are_additive=False)

    def restore(self):
        for owner,name,original in reversed(self.patches):
            setattr(owner,name,original)
        self.patches.clear()
