"""Plan-first, independent-process real-source H10 frozen benchmark. No training."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import inspect,json,random,resource,subprocess,sys,time
from copy import deepcopy
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from dstage_benchmark_support import VARIANTS,subset_inputs,BoundedStageProfiler
BASES={'original':'b291c645e0b821f2256ce9a5ee30f08e264cdb24',
       'round2':'3ff44d2ae4149f2488cb276042ee1ec77d72fc2c'}


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def check_source(root,expected=None):
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if expected and sha!=expected:raise ValueError('wrong actual baseline source SHA')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise ValueError('clean benchmark source required')
    return sha


def worker(a):
    # Select actual baseline modules BEFORE any project import.
    sys.path.insert(0,str(a.source_root.resolve()))
    import torch
    import pandas as pd
    from types import SimpleNamespace
    from analysis.finite_direct_continuation_state import safe_load
    import analysis.run_fc1_self_consistent_macro as fc1_runtime
    from analysis.run_fc1_self_consistent_macro import hashes
    from analysis.nested_equilibrium_utils import stable_json_hash
    from analysis.economic_config import AnalysisEconomicConfig
    from dataclasses import asdict as dataclass_dict
    from config.hyperparams import HyperParams
    import data.finite_direct_replay_runtime as runtime
    import data.finite_direct_initial_hstar as initial
    import data.simulate_ts_parallel as parallel
    from data.market_clearing import MarketClearingConfig,MarketClearingController
    from training.bp_simulation_policy import GridBPSimulationPolicy
    spec=VARIANTS[a.variant];sha=check_source(a.source_root,BASES.get(spec.source))
    for module in [runtime,initial,parallel]:
        if not Path(module.__file__).resolve().is_relative_to(a.source_root.resolve()):
            raise RuntimeError('baseline import contamination')
    if a.output.exists():raise ValueError('fresh independent output required')
    a.output.mkdir(parents=True)
    def forbidden(*args,**kwargs):raise RuntimeError('FROZEN_OPTIMIZER_FORBIDDEN')
    for cls in [torch.optim.Adam,torch.optim.AdamW,torch.optim.LBFGS]:cls.step=forbidden
    torch.Tensor.backward=forbidden;torch.set_num_threads(8)
    learning=safe_load(a.checkpoint);learning=learning.get('backend',learning)
    # The recorded initial root cache is READ ONLY and supplies configuration
    # and identity evidence only, never its processed state, action or labels.
    cache_identity=safe_load(a.initial_root_cache)['identity']
    actual_economic=cache_identity['economic']
    original_hp=fc1_runtime.build_hyperparams
    fc1_runtime.build_hyperparams=lambda:HyperParams(**actual_economic['hyperparameters'])
    try:
        models,episode,_,guard=fc1_runtime.setup(SimpleNamespace(device=a.device),initial_models=learning['transaction']['models'])
    finally:
        fc1_runtime.build_hyperparams=original_hp
    episode._fc1_firms=200;episode.refresh_firm_target();before=hashes(models)
    rebuilt=dict(economic=dataclass_dict(AnalysisEconomicConfig.from_current_config()),
                 hyperparameters=dataclass_dict(episode.hyperparams))
    if stable_json_hash(rebuilt)!=stable_json_hash(actual_economic):
        raise ValueError('recorded production economics/hyperparameters not reproduced')
    if before!=cache_identity['models']:
        raise ValueError('checkpoint models do not match recorded initial D-stage evidence')
    bank,schedule=subset_inputs(safe_load(a.bank),safe_load(a.schedule),a.parents)
    al_before=deepcopy(episode._get_sdf_al_state())
    rng_before=torch.random.get_rng_state().clone()
    cuda_rng_before=torch.cuda.get_rng_state_all() if torch.device(a.device).type=='cuda' else []
    options={'profile_root_timing':spec.timing_sync,'lightweight_hard_candidates':spec.light}
    root_options=dict(cache_identity['root'])
    root_options.update({k:v for k,v in options.items() if k in inspect.signature(MarketClearingConfig).parameters})
    config=MarketClearingConfig(**root_options)
    kwargs=dict(outer=12,stage='FROZEN_BENCHMARK',horizon=10,branches=2,capture=False,
                parent_batch_size=spec.batch,root_config=config)
    if 'reuse_shard_digest' in inspect.signature(runtime.generate_long_horizon_dataset).parameters:
        kwargs['reuse_shard_digest']=spec.digest_reuse
    signature={'fingerprint':'FROZEN_ROUND3_RECORDED_PARENTS_V1'}
    policies=[];original_init=GridBPSimulationPolicy.__init__
    def init(policy,*args,**kw):
        if spec.deferred_bp_stats:kw['defer_diagnostic_sync']=True
        original_init(policy,*args,**kw);policies.append(policy)
    GridBPSimulationPolicy.__init__=init
    warm_bank,warm_schedule=subset_inputs(bank,schedule,16)
    runtime.generate_long_horizon_dataset(models,episode,warm_bank,warm_schedule,signature,a.output/'warmup',**kwargs)
    policies.clear();device=torch.device(a.device)
    if device.type=='cuda':
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    counters=dict(lightweight_forward_actual_calls=0,Firm_forward_calls=0,
                  root_evaluation_calls=0,root_operator_evaluations=0,BP_grid_calls=0,
                  bytes_hashed=0,SHA_calls=0,explicit_cuda_synchronize_calls=0)
    if device.type=='cuda':
        original_sync=torch.cuda.synchronize
        def synchronize(*args,**kw):
            counters['explicit_cuda_synchronize_calls']+=1;return original_sync(*args,**kw)
        torch.cuda.synchronize=synchronize
    handles=[];seen=set()
    for model in [models['policy_value'],episode._target_policy_value()]:
        if id(model) in seen:continue
        seen.add(id(model))
        handles.append(model.register_forward_hook(lambda *args:counters.__setitem__('Firm_forward_calls',counters['Firm_forward_calls']+1)))
    for owner,name,key in [(MarketClearingController,'evaluate','root_evaluation_calls'),
            (GridBPSimulationPolicy,'evaluate_actions','BP_grid_calls')]:
        original=getattr(owner,name)
        def counted(*args,_original=original,_key=key,**kw):
            counters[_key]+=1
            if _key=='root_evaluation_calls':
                h=kw.get('h',args[4] if len(args)>4 else None)
                counters['root_operator_evaluations']+=h.numel()
            return _original(*args,**kw)
        setattr(owner,name,counted)
    if hasattr(MarketClearingController,'_hard_candidate_values'):
        original_light=MarketClearingController._hard_candidate_values
        def light(*args,**kw):
            counters['lightweight_forward_actual_calls']+=1;return original_light(*args,**kw)
        MarketClearingController._hard_candidate_values=light
    for module in [runtime,initial]:
        original_hash=module.file_hash
        def hashed(path,_original=original_hash):
            counters['SHA_calls']+=1;counters['bytes_hashed']+=Path(path).stat().st_size
            return _original(path)
        module.file_hash=hashed
    profiler=BoundedStageProfiler(device,max_regions=100000)
    if a.profile:
        for name,label in [('_process_node_batched','ECONOMIC_NODE_FULL'),('_apply_exit_batched','PARENT_EXIT'),
            ('_apply_entry_batched','CHILD_ENTRY'),('_predict_macro_fc1_batched','FC1_FORECAST')]:
            profiler.patch(parallel,name,label)
        profiler.patch(parallel,'_stable_transition_draws',lambda sim,state,child_t,branch_index:'CHILD%d_SHOCK'%branch_index)
        profiler.patch(runtime,'promote_child','NEXT_PARENT_PROMOTION')
        profiler.patch(GridBPSimulationPolicy,'__call__',lambda self,**kw:'PARENT_BP_GRID' if kw.get('bp_execution_applicable') else 'CHILD_BP_GRID')
        profiler.patch(runtime,'atomic_torch_save','PT_SERIALIZATION')
    activities=[torch.profiler.ProfilerActivity.CPU]
    if device.type=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
    started=time.monotonic()
    with ExitStack() as stack:
        trace=stack.enter_context(torch.profiler.profile(activities=activities,record_shapes=True)) if a.profile else None
        store=runtime.generate_long_horizon_dataset(models,episode,bank,schedule,signature,a.output/'measured',**kwargs)
        if device.type=='cuda':torch.cuda.synchronize(device)
        elapsed=time.monotonic()-started
    if trace:trace.export_chrome_trace(str(a.output/'torch_trace.json'))
    dump(a.output/'gpu_profile.json',profiler.finish() if a.profile else {'status':'NOT_PROFILED'})
    profiler.restore()
    frame=pd.concat(list(store.rows()),ignore_index=True)
    frame.sort_values(['parent_candidate_id','t','child_branch']).to_csv(a.output/'economic_rows.csv',index=False)
    metrics=[p.instrumentation() for p in policies]
    if hashes(models)!=before or guard['calls']:raise RuntimeError('model mutation or FC2 call')
    if episode._get_sdf_al_state()!=al_before:raise RuntimeError('AL state mutated')
    if not torch.equal(torch.random.get_rng_state(),rng_before):raise RuntimeError('global CPU RNG mutated')
    if any(not torch.equal(x,y) for x,y in zip(cuda_rng_before,torch.cuda.get_rng_state_all())):
        raise RuntimeError('global CUDA RNG mutated')
    sync_events=({e.key:e.count for e in trace.key_averages() if any(
        term in e.key.lower() for term in ['synchronize','_local_scalar_dense','aten::item'])}
        if trace else 'NOT_PROFILED')
    dump(a.output/'result.json',dict(variant=a.variant,source_commit=sha,configuration=asdict(spec),elapsed_seconds=elapsed,
        parents_per_second=a.parents/elapsed,valid_parent_times_per_second=store.manifest['sampling']['actual_parent_times']/elapsed,
        GPU_peak_memory_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
        CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        RSS_scope='INDEPENDENT_PROCESS_INCLUDING_WARMUP',shard_count=len(store.manifest['shards']),
        bytes_written=sum(f.stat().st_size for f in (a.output/'measured').rglob('*') if f.is_file()),counters=counters,
        BP_candidate_evaluations=sum(m['num_candidate_evaluations'] for m in metrics),model_hashes=before,
        models_unchanged=True,optimizer_steps=0,AL_state_unchanged=True,global_RNG_unchanged=True,
        GPU_CPU_sync_counts=sync_events,Firm_forward_calls_scope='WHOLE_MODEL_FORWARD_ONLY',
        bank_fingerprint=bank['fingerprint'],schedule_fingerprint=schedule['fingerprint'],
        crn_identity=stable_json_hash(dict(seed=schedule['seed'],parent_ids=bank['state']['transition_path_id'].tolist())),
        profiling_enabled=a.profile,full_path_economic_parity='NOT_TESTED',
        input_hashes={k:initial.file_hash(getattr(a,k)) for k in ['checkpoint','bank','schedule','initial_root_cache']}))
    for handle in handles:handle.remove()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint','bank','schedule','initial-root-cache','output','source-original','source-round2']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parents',type=int,choices=[128,512,1024],default=128)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--device',default='cuda:0')
    p.add_argument('--execute-frozen',action='store_true');p.add_argument('--profile',action='store_true')
    p.add_argument('--variant',choices=VARIANTS,help=argparse.SUPPRESS)
    p.add_argument('--source-root',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.variant:
        if not a.execute_frozen or a.source_root is None:raise ValueError('explicit frozen worker/source required')
        if a.variant=='COMBINED_OPTIMIZATION':raise ValueError('combined blocked pending full-path parity certification')
        return worker(a)
    if a.repeats<1:raise ValueError('positive repeats required')
    names=[n for n in VARIANTS if n!='COMBINED_OPTIMIZATION'];random.Random(20261022).shuffle(names)
    if a.output.exists():raise ValueError('fresh output required')
    a.output.mkdir(parents=True)
    dump(a.output/'benchmark_plan.json',dict(variants={n:asdict(VARIANTS[n]) for n in names},order=names,
        parents=a.parents,repeats=a.repeats,warmup_parents=16,independent_processes=True,formal_training=False,
        combined='BLOCKED_PENDING_FULL_PATH_PARITY',dynamic_transition='NOT_IMPLEMENTED',
        filesystem_cache='REPORT_ORDER_NOT_COLD_CACHE',transfer='BLOCKED_TRANSFER_PERMISSION'))
    if not a.execute_frozen:return
    roots=dict(current=HERE.parent,original=a.source_original,round2=a.source_round2)
    for repeat in range(a.repeats):
        for name in names:
            cmd=[sys.executable,str(Path(__file__).resolve()),'--execute-frozen','--variant',name,
                '--source-root',str(roots[VARIANTS[name].source]),'--source-original',str(a.source_original),
                '--source-round2',str(a.source_round2),'--parents',str(a.parents),'--device',a.device,
                '--output',str(a.output/('%s_repeat%d'%(name,repeat)))]
            for key in ['checkpoint','bank','schedule','initial_root_cache']:
                cmd+=['--'+key.replace('_','-'),str(getattr(a,key))]
            if a.profile:cmd+=['--profile']
            result=subprocess.run(cmd,check=False)
            if result.returncode:
                dump(a.output/('%s_FAILURE.json'%name),dict(returncode=result.returncode,status='NOT_ACCEPTED'))
                raise RuntimeError('worker failed; outputs preserved, stop comparisons')


if __name__=='__main__':main()
