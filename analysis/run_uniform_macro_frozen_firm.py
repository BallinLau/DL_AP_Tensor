"""Explicitly authorized Firm-only solve on wide uniform macro collocation.

No simulation, root search, FC1 fit, SDF fit or equilibrium Outer is executed.
Reuse production P/Q/BP blocks and StageJournal, never the full Outer driver.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import torch
from analysis.finite_direct_continuation_state import safe_load
from analysis.finite_direct_outer4_state import StageJournal
from analysis.nested_equilibrium_utils import (atomic_torch_save,atomic_write_json,atomic_write_bytes,
    restore_training_transaction,stable_json_hash,module_hash)
from analysis.run_full_nested_finite_direct_outer4 import ProductionBackend,restore_fresh
from analysis.run_fc1_self_consistent_macro import hashes
from copy import deepcopy
from data.finite_direct_runtime import conditional_child_bank,check_firm_lineage
from data.finite_direct_uniform_firm_macro import uniform_macro_bank,sampling_spec,FirmMacroCoverageConfig
from dataclasses import asdict
from data.finite_direct_sampling import file_hash
from training.inner_fixed_point import InnerFPCollocationBank

FIRM_CFG=FirmMacroCoverageConfig(firm_macro_coverage_enabled=True,firm_macro_coverage_mode='STANDALONE',firm_macro_coverage_fraction=1.)

def plan():
    result=dict(experiment='uniform_macro_frozen_firm_20261021',scope='FIRM_ONLY',
        sampling=asdict(FIRM_CFG),train_rows=FIRM_CFG.firm_macro_sampling_budget,validation_rows=min(2048,FIRM_CFG.firm_macro_sampling_budget),firm_batch_size=FIRM_CFG.firm_macro_batch_size,
        firm_inner_max=3,p_policy_outer_max=3,p_adam_steps=200,p_lbfgs_max_iter=100,
        q_picard_outer_max=3,q_fit_steps=300,policy_distill_max_steps=1000,
        frozen=['fc1_c','fc1_k','sdf','wealth'],macro_optimizer_steps=0,fc2=False,root_search=False,
        source='JOB702636_OUTER11_MODEL_AND_MICRO_DONORS',
        rng_seed=20261022,resume='RESTART_UNCOMMITTED_FIRM_BLOCK',
        optimizer_semantics='PRODUCTION_BLOCK_LOCAL_P_Q_OPTIMIZERS; SOURCE_PERSISTENT_BP_STATE_RESTORED')
    return result


def blocks():
    yield 'initial_metrics',None,{}
    for inner in range(3):
        for policy in range(3):yield f'inner_{inner:02d}/P_{policy:02d}','P',dict(inner=inner,policy=policy)
        for picard in range(3):yield f'inner_{inner:02d}/Q_{picard:02d}','Q',dict(inner=inner,picard=picard)
        yield f'inner_{inner:02d}/BP','BP',dict(inner=inner)
        yield f'inner_{inner:02d}/metrics',None,dict(inner=inner)


def verify_sources(args):
    manifest=json.loads(args.source_manifest.read_text())
    for name,digest in manifest['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or file_hash(path)!=digest:
            raise ValueError(f'source snapshot mismatch: {name}')
    expected=dict(schema='FIRM_ONLY_UNIFORM_MACRO_APPROVAL_V1',approved=True,
        plan_fingerprint=stable_json_hash(plan()),source_manifest_sha256=file_hash(args.source_manifest),
        initial_checkpoint_sha256=file_hash(args.initial_checkpoint),
        donor_bank_sha256=file_hash(args.donor_bank))
    if args.resume_from_run is not None:
        pointer=args.resume_from_run/'transactions/latest.json'
        previous=json.loads(pointer.read_text())
        expected['import_committed_source']=dict(run=str(args.resume_from_run.resolve()),
            pointer_sha256=file_hash(pointer),checkpoint_sha256=previous['sha256'])
    actual=json.loads(args.authorization.read_text())
    if any(actual.get(k)!=v for k,v in expected.items()):raise ValueError('Firm-only authorization mismatch')
    return expected


def history_rows(results):
    # Production stage metrics already contain their short kind in 'stage'.
    # Keep that separately and use the journal identifier as the unique index.
    return [dict(r,stage_kind=r.get('stage'),stage=name) for name,r in results.items()]


def import_committed_stages(journal,old,source_run):
    expected=[name for name,_,_ in blocks()]
    if old['completed']!=expected[:len(old['completed'])]:raise ValueError('not a Firm-only committed prefix')
    for name in ('plan_fingerprint','initial_checkpoint_sha256','donor_bank_sha256'):
        if old['contract'][name]!=journal.contract[name]:raise ValueError(f'import contract changed: {name}')
    journal.backend.restore(old['backend'])
    journal.completed=list(old['completed']);journal.results=deepcopy(old['results'])
    journal._publish('IMPORTED_COMMITTED_FIRM_STAGES',dict(source_run=str(source_run),
        source_contract=old['contract'],imported_stages=journal.completed,
        reason='LOG_EXPORT_DUPLICATE_STAGE_KEY_ONLY_NO_OPERATOR_CHANGE'))


def guard_macro_optimizers(backend):
    forbidden_ids={id(p) for p in backend.models['sdf_fc1'].parameters()}
    guarded=[]
    def forbidden(*a,**k):raise RuntimeError('Macro optimizer forbidden in Firm-only run')
    for name,opt in {**backend.episode.optimizers,'fc1_conditional':backend.fc1}.items():
        if any(id(p) in forbidden_ids for group in opt.param_groups for p in group['params']):
            opt.step=forbidden;guarded.append(name)
    return guarded


class FirmBackend(ProductionBackend):
    def _conditional(self,bank,seed):
        return conditional_child_bank(self.models['sdf_fc1'],bank,seed=seed,n_child_shocks=2,
            economic=self.economic,device=self.episode.device,chunk_size=self.args.firm_batch_size,
            m_min=self.episode.hyperparams.pv_m_clamp_min,m_max=self.episode.hyperparams.pv_m_clamp_max)


def run(args):
    binding=verify_sources(args)
    args.firm_batch_size=FIRM_CFG.firm_macro_batch_size;args.parent_batch_size=128;args.fit_parent_batch_size=128
    args.initial_hstar_h50_adapter=False
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    learning=safe_load(args.initial_checkpoint)
    if not {'transaction','fresh','state'}<=set(learning):raise ValueError('full learning state required')
    backend=FirmBackend(args,({},{},dict(fingerprint=stable_json_hash(plan()))),learning_snapshot=learning)
    restore_training_transaction(learning['transaction'],models=backend.models,
        optimizers={**backend.episode.optimizers,'fc1_conditional':backend.fc1},episode=backend.episode)
    restore_fresh(backend.episode,learning['fresh']);backend.episode.refresh_firm_target()
    frozen={k:v for k,v in hashes(backend.models).items() if k!='firm'}
    atomic_write_json(args.output/'metadata/config.json',plan())
    atomic_write_json(args.output/'metadata/input_binding.json',binding)
    atomic_write_json(args.output/'metadata/frozen_initial_hashes.json',frozen)
    guarded=guard_macro_optimizers(backend)
    source=safe_load(args.donor_bank)
    banks={};children={}
    for split,offset in [('train',0),('validation',100000)]:
        p=source['banks'][split]
        donor=InnerFPCollocationBank(**{k:v for k,v in p.items() if k!='fingerprint'})
        banks[split]=uniform_macro_bank(donor,seed=FIRM_CFG.firm_macro_sampling_seed+offset,
            hatcf_bounds=FIRM_CFG.firm_macro_h_bounds,lnkf_bounds=FIRM_CFG.firm_macro_ell_bounds)
        if len(banks[split].parent_states)!=plan()[f'{split}_rows']:
            raise ValueError('unexpected source donor budget')
        children[split]=backend._conditional(banks[split],20261023+offset)
        if not torch.isfinite(children[split].children).all():raise FloatingPointError('nonfinite conditional children')
        if not torch.isfinite(children[split].metadata['raw_m']).all():raise FloatingPointError('nonfinite raw M')
    check_firm_lineage(banks['train'],banks['validation'])
    path=args.output/'outer_000/firm_conditional_bank.pt'
    if not path.exists():
        atomic_torch_save(path,dict(banks={s:b.payload() for s,b in banks.items()},
            children={s:b.payload() for s,b in children.items()},macro_hash=module_hash(backend.models['sdf_fc1'])))
    else:
        old=safe_load(path)
        for s in banks:
            if old['banks'][s]['fingerprint']!=banks[s].fingerprint or not torch.equal(old['children'][s]['children'],children[s].children):
                raise ValueError('uniform bank/CRN changed on resume')
    backend.state['firm_banks']['0']=dict(path=str(path),sha256=file_hash(path))
    backend.state['probe_bank']=dict(path=str(path),sha256=file_hash(path))
    coverage={s:dict(rows=len(b.parent_states),Hatcf_min=float(b.parent_states[:,5].min()),
        investment_cost_min=float(b.parent_states[:,3].min()),
        investment_cost_max=float(b.parent_states[:,3].max()),
        child_cost_max=float(children[s].children[:,:,3].max()),
        Hatcf_max=float(b.parent_states[:,5].max()),LnKF_min=float(b.parent_states[:,6].min()),
        LnKF_max=float(b.parent_states[:,6].max()),parent_fingerprint=b.fingerprint,
        child_fingerprint=children[s].fingerprint) for s,b in banks.items()}
    atomic_write_json(args.output/'metadata/bank_coverage.json',coverage)
    print(json.dumps(dict(event='BANK_READY',coverage=coverage,guarded_macro_optimizers=guarded)),flush=True)
    journal=StageJournal(args.output/'transactions',binding,backend,resume=args.resume)
    if args.resume_from_run is not None:
        old_pointer=json.loads((args.resume_from_run/'transactions/latest.json').read_text())
        old_checkpoint=args.resume_from_run/'transactions'/old_pointer['checkpoint']
        if file_hash(old_checkpoint)!=old_pointer['sha256']:raise ValueError('import checkpoint corruption')
        old=safe_load(old_checkpoint)
        import_committed_stages(journal,old,args.resume_from_run)
        if any(hashes(backend.models)[k]!=v for k,v in frozen.items()):raise RuntimeError('imported Macro changed')
        atomic_write_json(args.output/'metadata/committed_import.json',binding['import_committed_source'])
    def execute(name,kind,indices):
        before=hashes(backend.models)
        print(json.dumps(dict(event='STAGE_START',stage=name,hashes=before)),flush=True)
        result=(backend.execute(0,kind,indices) if kind else backend._cross(0,label=name.replace('/','_')))
        after=hashes(backend.models)
        if any(after[k]!=v for k,v in frozen.items()):raise RuntimeError('frozen Macro hash changed')
        if backend.fc2_guard['calls'] or any(backend.counter.values()):raise RuntimeError('root/FC2 guard violated')
        result.update(macro_frozen_verified=True,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
            elapsed_seconds=time.monotonic()-started)
        print(json.dumps(dict(event='STAGE_COMPLETE',stage=name,result=result),default=str),flush=True)
        return result
    for name,kind,indices in blocks():
        journal.run(name,lambda n=name,k=kind,i=indices:execute(n,k,i))
        rows=history_rows(journal.results)
        pd.json_normalize(rows).to_csv(args.output/'firm_training_history.csv',index=False)
    final=backend.snapshot()
    atomic_torch_save(args.output/'firm_complete.pt',final)
    summary=dict(status='COMPLETED',training_performed=True,firm_inner_completed=3,
        completed_stages=journal.completed,initial_metrics=journal.results['initial_metrics'],
        final_metrics=journal.results['inner_02/metrics'],macro_frozen=True,macro_optimizer_steps=0,
        final_hashes=hashes(backend.models),peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
        checkpoint_sha256=file_hash(args.output/'firm_complete.pt'),plan=plan(),
        source_binding=binding,elapsed_seconds=time.monotonic()-started)
    atomic_write_json(args.output/'summary.json',summary)
    atomic_write_bytes(args.output/'report.md',('# Frozen Macro / uniform Firm training\n\n'
        'Completed three Firm inner cycles; no FC1/SDF/Wealth optimizer steps.\n\n'
        'See summary.json, firm_training_history.csv and transactions/ for actual residuals and optimizer logs.\n'
        'This is conditional Firm training, not a demonstrated macro equilibrium.\n').encode())
    print(json.dumps(dict(event='TRAINING_COMPLETE',summary=str(args.output/'summary.json'))),flush=True)


def main(argv=None):
    global FIRM_CFG
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--initial-checkpoint',type=Path)
    p.add_argument('--donor-bank',type=Path)
    p.add_argument('--source-manifest',type=Path)
    p.add_argument('--authorization',type=Path)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--resume-from-run',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--firm-macro-config',type=Path)
    args=p.parse_args(argv)
    if args.firm_macro_config is not None:FIRM_CFG=FirmMacroCoverageConfig(**json.loads(args.firm_macro_config.read_text()))
    FIRM_CFG.validate()
    if not FIRM_CFG.firm_macro_coverage_enabled or FIRM_CFG.firm_macro_coverage_mode!='STANDALONE':p.error('Firm-only runner requires explicit STANDALONE coverage')
    if args.resume and args.resume_from_run is not None:p.error('choose intra-run resume OR committed import')
    if not args.execute:
        atomic_write_json(args.output/'plan.json',plan());return 0
    if any(getattr(args,k) is None for k in ('initial_checkpoint','donor_bank','source_manifest','authorization')):
        p.error('explicit authorized source artifacts required')
    if not args.device.startswith('cuda'):p.error('training requires the authorized GPU allocation')
    torch.set_num_threads(8)
    run(args)
    return 0


if __name__=='__main__':raise SystemExit(main())
