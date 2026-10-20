#!/usr/bin/env python3
"""Review-gated CROSS_RUN_CONTINUATION_FROM_COMMITTED_OUTER3.

Default: plan-only. No backend or optimizer is constructed without dynamic v4.
--resume means ONLY intra-run recovery of this new continuation run.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.run_full_nested_finite_direct_outer4 import (ProductionBackend,Budget,stages as old_stages,
    stage_id,restore_fresh,set_scope)
from analysis.finite_direct_outer4_state import StageJournal
from analysis.finite_direct_continuation_state import (RUN_ID,SOURCE_SHA,EXPECTED_HASHES,safe_load,
    state_digest,component_hashes,new_backend_state,verify_optimizer_restored,verify_economic_contract)
from analysis.finite_direct_continuation_probes import (make_macro_probe,evaluate_fixed,save_movement,write_probe_histories)
from analysis.nested_equilibrium_utils import (atomic_write_json,atomic_torch_save,stable_json_hash,
    restore_training_transaction,capture_rng_state,restore_rng_state,atomic_write_bytes)
from data.finite_direct_runtime import load_locked_sampling,generate_dataset,firm_bank_from_captures
from data.finite_direct_sampling import file_hash
from analysis.finite_direct_outer_metrics import economic_metrics
from analysis.run_fc1_self_consistent_macro import evaluate
from analysis.finite_direct_dynamic_approval import approval_binding,require_approval as require_continuation_approval
from data.finite_direct_dynamic_empirical import (CONFIG as DYNAMIC_CONFIG,initial_pool,next_pool,resample_train,save_bundle,load_bundle,source_summary)
from data.finite_direct_runtime import OPERATOR
from training.inner_fixed_point import InnerFPCollocationBank


def stages():
    yield 3,'dynamic_initialize',{}
    yield 3,'baseline',{}
    for outer,kind,indices in old_stages(replace(Budget(),equilibrium_outer_max=12)):
        if outer>=4:
            if kind=='cross_block':
                yield outer,'empirical_source',{}
                yield outer,'EMPIRICAL_POOL_COMMIT',{}
            yield outer,kind,indices


def training_plan():
    return dict(experiment=RUN_ID+'_dynamic_exit_v2',initialization='CROSS_RUN_CONTINUATION_FROM_COMMITTED_OUTER3',
        intra_run_resume='NEW_STAGEJOURNAL_ONLY',start_outer=4,end_outer=11,outers_to_execute=8,
        budget=asdict(replace(Budget(),equilibrium_outer_max=8)),train_parents=8192,validation_parents=2048,
        firms=200,children=2,beta_empirical_ratio='50:50',operator=OPERATOR,
        empirical_semantics='DYNAMIC_TRAIN_ONLY_REALIZED_PRE_DECISION_B',dynamic_sampling=DYNAMIC_CONFIG,
        source_result_limit='OLD_OPERATOR_LEARNING_STATE_WARM_START_NOT_CONVERGENCE_EVIDENCE',root_search=False,fc2=False,
        execution=[dict(id=stage_id(o,k,i),outer=o,kind=k,indices=i) for o,k,i in stages()],
        baseline='OUTER3_ON_NEW_VALIDATION_BANK_BASELINE',physical_batch=dict(parent=128,fc1=128,firm=512),
        convergence_is_report_only=True,retention='KEEP_ALL_TRANSACTIONS_NO_AUTOMATIC_DELETION')


def verify_initial(path,manifest_path,signature):
    source=json.loads(Path(manifest_path).read_text()); payload=safe_load(path)
    if payload.get('format')!='finite_direct_cross_run_continuation_v1': raise ValueError('weights-only continuation is forbidden')
    if source.get('source_terminal_checkpoint_sha256')!=SOURCE_SHA or source.get('status')!='PASS':
        raise ValueError('uncertified continuation source')
    if source.get('source_terminal_stage')!='outer_003/outer_commit': raise ValueError('wrong continuation stage')
    if payload['new_sampling_signature']!=signature: raise ValueError('continuation bank mismatch')
    learning=payload['learning']
    if component_hashes(learning['transaction']['models'])!=EXPECTED_HASHES: raise ValueError('initial model mismatch')
    if state_digest(learning)!=payload['learning_sha256'] or state_digest(learning)!=source['learning_sha256']:
        raise ValueError('learning state digest mismatch')
    if learning['source_sdf_epoch']!=400 or learning['source_outer']!=3: raise ValueError('wrong learning counters')
    if learning['transaction']['episode_runtime']['sdf_al_state']!=source['source_al_state']:
        raise ValueError('initial AL mismatch')
    for name,audit in source['optimizers'].items():
        if state_digest(learning['transaction']['optimizers'][name])!=audit['sha256']:
            raise ValueError(f'initial optimizer mismatch: {name}')
    if state_digest(learning['fresh'])!=source['source_fresh_hash']: raise ValueError('initial fresh RNG mismatch')
    return payload,source


class ContinuationBackend(ProductionBackend):
    def __init__(self,args,locked,initial,source):
        learning=initial['learning']
        super().__init__(args,locked,learning_snapshot=learning)
        economic_audit=verify_economic_contract(learning['source_economic_contract'],
            dict(economic=asdict(self.economic),hyperparameters=asdict(self.episode.hyperparams)))
        restore_training_transaction(learning['transaction'],models=self.models,
            optimizers={**self.episode.optimizers,'fc1_conditional':self.fc1},episode=self.episode)
        restore_fresh(self.episode,learning['fresh'])
        self.state=new_backend_state(learning)
        self.state['economic_import_audit']=economic_audit
        self.state['optimizer_import_audit']={name:verify_optimizer_restored(
            learning['transaction']['optimizers'][name],opt) for name,opt in
            {**self.episode.optimizers,'fc1_conditional':self.fc1}.items()}
        self.episode.refresh_firm_target()
        if component_hashes({k:m.state_dict() for k,m in self.models.items()})!=source['model_hashes']:
            raise ValueError('restored live model mismatch')
        # Archive source RNG/bank intact in initial checkpoint. Active capacity
        # belongs to NEW parent IDs. Production creates/reseeds at global epoch
        # 400, using its unchanged bank/paired-shock rule (not repeated old eps).
        self.episode._sdf_shock_bank=None; self.episode._sdf_shock_bank_key=None
        self.episode._sdf_shock_bank_n_parents=0
        self.state['fresh_rebinding']=dict(source_sdf_epoch=400,
            source_sdf_optimizer_steps=source['source_sdf_optimizer_steps'],source_al_hash=source['source_al_hash'],
            source_pair_rng_hash=state_digest(learning['fresh']['pair_rng']),
            new_fresh_shock_bank_version=self.signature['fingerprint'],
            new_fresh_shock_rng_identity='production episode_id + global epoch (400 onward) + new capacity + dtype/device',
            new_sdf_parent_index_range='0 .. legal_train_pairs + legal_validation_pairs - 1',
            bitwise_old_run_counterfactual=False)
        self.state['function_snapshots']={}
        self.state['run_identity']=training_plan()['experiment']
        self.state['new_code_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        self._initial_train=self.banks['train'];self._initial_schedule=self.schedules['train']
        self._validation_identity=(file_hash(args.sampling_dir/'validation_initial_bank.pt'),
            stable_json_hash(self.schedules['validation']))

    def restore(self,snapshot):
        super().restore(snapshot)
        if 'active_dynamic_outer' in self.state:
            self._activate_dynamic(self.state['active_dynamic_outer'])

    def _activate_dynamic(self,outer):
        record=self.state['dynamic_banks'][str(outer)]
        _,bank,schedule=load_bundle(record)
        if record['outer']!=outer: raise ValueError('wrong committed dynamic Outer')
        self.banks['train']=bank;self.schedules['train']=schedule
        self.state['active_dynamic_outer']=outer
        if (file_hash(self.args.sampling_dir/'validation_initial_bank.pt'),stable_json_hash(self.schedules['validation']))!=self._validation_identity:
            raise ValueError('fixed Validation changed')

    def _dynamic_initialize(self):
        path=self.args.sampling_dir/'initial_dynamic_pool.pt'
        pool=safe_load(path)
        expected=initial_pool(safe_load(self.args.sampling_dir/'empirical_template_bank.pt'))
        if state_digest(pool)!=state_digest(expected): raise ValueError('unreviewed initial empirical pool')
        bank,schedule,history=resample_train(self._initial_train,self._initial_schedule,pool,outer=4)
        record=save_bundle(self.root/'dynamic/outer_004',pool,bank,schedule,history)
        self.state['dynamic_banks']={'4':record};self._activate_dynamic(4)
        return dict(initialization='LOCKED_JOB699894_T0',pool_sha=file_hash(path),optimizer_steps=0)

    def _empirical_source(self,outer):
        set_scope(self.models,'frozen');rng=capture_rng_state();start=time.monotonic()
        before=component_hashes({k:m.state_dict() for k,m in self.models.items()})
        try:
            path=self.root/f'outer_{outer:03d}'/'POST_SDF_TRAIN_EMPIRICAL_SOURCE'
            with torch.inference_mode():
                _,_,manifest=generate_dataset(self.models,self.episode,self.banks['train'],self.schedules['train'],
                    self.signature,path,outer=outer,stage=DYNAMIC_CONFIG['source'],
                    parent_batch_size=self.args.parent_batch_size,capture=False,predecision_templates=True)
        finally: restore_rng_state(rng)
        if before!=component_hashes({k:m.state_dict() for k,m in self.models.items()}): raise ValueError('source changed model')
        self.state.setdefault('dynamic_sources',{})[str(outer)]=dict(path=str(path),manifest_sha256=file_hash(path/'manifest.json'),model_hashes=before)
        return dict(optimizer_steps=0,source_seconds=time.monotonic()-start,
            source_stage=DYNAMIC_CONFIG['source'],predecision_template_count=manifest['captures'],
            peak_gpu_memory_gib=torch.cuda.max_memory_allocated(self.episode.device)/2**30 if self.episode.device.type=='cuda' else 0)

    def _pool_commit(self,outer):
        start=time.monotonic();record=self.state['dynamic_sources'][str(outer)];path=Path(record['path'])
        if file_hash(path/'manifest.json')!=record['manifest_sha256']: raise ValueError('source manifest changed')
        manifest=json.loads((path/'manifest.json').read_text());captures=[]
        for name,digest in manifest['shards'].items():
            if Path(name).name!=name or file_hash(path/name)!=digest: raise ValueError('source shard changed')
            captures.extend(safe_load(path/name)['captures'])
        pool,_,_=load_bundle(self.state['dynamic_banks'][str(outer)])
        result=next_pool(pool,captures,source_outer=outer,source_hashes=record['model_hashes'],source_manifest_sha=record['manifest_sha256'])
        stats=dict(outer=outer,source_outer=outer,source_model_hashes=record['model_hashes'],
            pool_update_seconds=time.monotonic()-start,recent_templates=len(result['recent']),archive_templates=len(result['archive']),
            unique_source_parents=len({t['cluster_id'] for t in result['recent']}),
            pool_fingerprint=result['fingerprint'],source_failures=len(result['failures']))
        stats.update(source_summary(result['recent']))
        if outer<11:
            bank_start=time.monotonic()
            bank,schedule,history=resample_train(self._initial_train,self._initial_schedule,result,outer=outer+1)
            bundle=save_bundle(self.root/f'dynamic/outer_{outer+1:03d}',result,bank,schedule,history)
            self.state['dynamic_banks'][str(outer+1)]=bundle
            stats.update(bank_construction_seconds=time.monotonic()-bank_start,
                artifact_bytes=sum(Path(bundle[k]['path']).stat().st_size for k in ('pool','bank','schedule')),
                bank_fingerprint=bundle['bank_fingerprint'],
                unique_sampled_source_parent_count=len({h['cluster_id'] for h in history}),
                recent_parent_coverage=len({h['cluster_id'] for h in history if h['pool_source']=='RECENT'})/stats['unique_source_parents'],
                template_reuse_fraction=1-len({h['template_id'] for h in history})/len(history),
                conditional_fallback_fraction=sum(h['fallback_fraction'] for h in history)/len(history))
        else:
            # Persist final distributions, but never construct nonexistent Bank12.
            atomic_write_json(self.root/'dynamic/final_empirical_pool_statistics.json',result)
        stats.update(update_seconds=time.monotonic()-start,optimizer_steps=0)
        self.state.setdefault('dynamic_history',{})[str(outer)]=stats
        return stats

    def _fixed_bank(self):
        record=self.state['probe_bank']; path=Path(record['path'])
        if file_hash(path)!=record['sha256']: raise ValueError('fixed Firm probe changed')
        payload=safe_load(path); bank=payload['banks']['validation']
        return InnerFPCollocationBank(**{k:v for k,v in bank.items() if k!='fingerprint'})

    def _record_functions(self,outer):
        record=self.state['macro_probe']; path=Path(record['path'])
        if file_hash(path)!=record['sha256']: raise ValueError('fixed macro probe changed')
        current=evaluate_fixed(self,self._fixed_bank(),safe_load(path))
        snapshots=self.state['function_snapshots']
        def prior(index):
            record=snapshots.get(str(index))
            if record is None: return None
            if file_hash(record['path'])!=record['sha256']: raise ValueError('previous probe output changed')
            return safe_load(record['path'])
        rows=save_movement(self.root,outer,current,prior(outer-1),prior(outer-2))
        path=self.root/f'outer_{outer:03d}'/'fixed_function_outputs.pt'
        atomic_torch_save(path,current)
        snapshots[str(outer)]=dict(path=str(path),sha256=file_hash(path))
        return rows

    def _baseline(self):
        split='validation'; stage='OUTER3_ON_NEW_VALIDATION_BANK_BASELINE'
        set_scope(self.models,'frozen'); rng=capture_rng_state()
        try:
            path=self.root/stage/split
            frame,captures,manifest=generate_dataset(self.models,self.episode,self.banks[split],self.schedules[split],
                self.signature,path,outer=3,stage=stage,parent_batch_size=self.args.parent_batch_size,capture=True)
            bank=firm_bank_from_captures(captures,split=split,max_rows=512,seed=202610204)
            child=self._conditional(bank,bank.seed+1)
            probe_path=self.root/'immutable_artifacts/fixed_firm_probe.pt'
            atomic_torch_save(probe_path,dict(banks={split:bank.payload()},children={split:child.payload()}))
            self.state['probe_bank']=dict(path=str(probe_path),sha256=file_hash(probe_path))
            macro_path=self.root/'immutable_artifacts/fixed_macro_probe.pt'
            atomic_torch_save(macro_path,make_macro_probe(frame))
            self.state['macro_probe']=dict(path=str(macro_path),sha256=file_hash(macro_path))
            functions=self._record_functions(3)
        finally: restore_rng_state(rng)
        return dict(evaluation=stage,optimizer_steps=0,function_movement=functions,
            economic_metrics=economic_metrics(frame,self.schedules[split],outer=3,stage=stage),
            manifest_sha256=file_hash(path/'manifest.json'))

    def _execute(self,outer,kind,i):
        start=time.monotonic()
        if kind=='dynamic_initialize': result=self._dynamic_initialize()
        elif kind=='empirical_source': result=self._empirical_source(outer)
        elif kind=='EMPIRICAL_POOL_COMMIT': result=self._pool_commit(outer)
        elif kind=='baseline': result=self._baseline()
        elif kind=='outer_commit':
            movement_rows=self._record_functions(outer)
            result=super()._execute(outer,kind,i); result['function_movement']=movement_rows
        else:
            if kind=='outer_start': self._activate_dynamic(outer)
            if outer>=4 and kind not in ('outer_start',) and self.state.get('active_dynamic_outer')!=outer:
                raise ValueError('current Outer bank not activated')
            result=super()._execute(outer,kind,i)
        result['stage_wallclock_seconds']=time.monotonic()-start
        return result

    def _fc1_epoch(self,outer,i):
        pre=None
        if i['epoch']==1:
            pre=evaluate(self.models['sdf_fc1'],self._data(outer,i['macro'],'validation'),self.episode.device)
        result=super()._fc1_epoch(outer,i)
        if pre is not None: result['pre_fit_validation']=pre
        if 'validation' in result: result['post_fit_validation']=result['validation']
        return result

    def _sdf_chunk(self,outer,chunk):
        result=super()._sdf_chunk(outer,chunk)
        bank=self.episode._sdf_shock_bank
        result.update(new_sdf_parent_index_range=[0,int(bank.eps.shape[0])-1],
            new_fresh_shock_bank_version=self.signature['fingerprint'],
            new_fresh_shock_rng_identity=dict(base_seed=bank.base_seed,refresh_id=bank.refresh_id,
                production_key=list(self.episode._sdf_shock_bank_key)),
            global_sdf_epoch=self.state['sdf_epoch'])
        self.state['fresh_rebinding']['last_committed_binding']={k:result[k] for k in (
            'new_sdf_parent_index_range','new_fresh_shock_bank_version','new_fresh_shock_rng_identity','global_sdf_epoch')}
        return result

    def write_histories(self,results,*,figures=False):
        super().write_histories(results,figures=figures)
        write_probe_histories(self.root,results,figures=figures)
        from analysis.finite_direct_dynamic_diagnostics import write_dynamic_histories
        write_dynamic_histories(self,figures=figures)
        training=pd.read_csv(self.root/'outer_training_history.csv')
        mask=training.get('probe_kind',pd.Series(index=training.index,dtype=object)).notna()
        training[mask].to_csv(self.root/'firm_residual_history.csv',index=False)
        economic=pd.read_csv(self.root/'outer_macro_economic_history.csv')
        columns=[c for c in economic if any(x in c for x in ('count','fraction')) or c in ('outer','stage','split','sampling_source')]
        economic[columns].to_csv(self.root/'parent_child_validity_history.csv',index=False)
        fit=[]
        for r in results.values():
            if r.get('stage')!='FC1': continue
            for phase in ('pre_fit_validation','post_fit_validation'):
                if phase in r: fit.append(dict(outer=r['outer'],macro_inner=r['indices']['macro'],
                    epoch=r['indices']['epoch'],evaluation=phase,**r[phase]))
        pd.DataFrame(fit).to_csv(self.root/'fc1_fit_history.csv',index=False)
        common=dict(source_run='Job701387',source_commit=source_commit(),source_outer=3,
            economic_operator=OPERATOR,historical_results_are_corrected_convergence_evidence=False,
            new_run=training_plan()['experiment'],new_commit=self.state['new_code_commit'],sampling_signature=self.signature['fingerprint'],
            validation_crn_hash=stable_json_hash([r['child_shock_keys'] for r in self.schedules['validation']['parents']]))
        for filename in ('outer_training_history.csv','outer_macro_economic_history.csv','outer_sdf_al_history.csv',
            'firm_function_movement.csv','fc1_function_movement.csv','sdf_wealth_function_movement.csv',
            'firm_residual_history.csv','parent_child_validity_history.csv','outer_execution_timing.csv','fc1_fit_history.csv'):
            path=self.root/filename
            try: frame=pd.read_csv(path)
            except pd.errors.EmptyDataError: frame=pd.DataFrame()
            for key,value in common.items():frame[key]=value
            frame.to_csv(path,index=False,na_rep='NOT_AVAILABLE')


def source_commit():
    from analysis.finite_direct_continuation_state import SOURCE_COMMIT
    return SOURCE_COMMIT


def install_historical_baseline(prepared,output,source,*,resume=False):
    original=Path(prepared)/'historical_baseline';target=Path(output)/'historical_baseline'
    hashes=source.get('historical_artifacts')
    if not hashes: raise ValueError('historical baseline content manifest missing')
    for name,digest in hashes.items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts: raise ValueError('unsafe historical artifact path')
        if file_hash(original/rel)!=digest: raise ValueError('historical artifact changed')
    if resume:
        for name,digest in hashes.items():
            if file_hash(target/name)!=digest: raise ValueError('new-run read-only history changed')
    else:
        for name,digest in hashes.items():
            destination=target/name
            if destination.exists():
                if file_hash(destination)!=digest: raise ValueError('partial initialization history differs')
            else: atomic_write_bytes(destination,(original/name).read_bytes())
    return dict(copied_historical_files=len(hashes),optimizer_steps=0)


def drive(backend,journal):
    for outer,kind,indices in stages():
        journal.run(stage_id(outer,kind,indices),lambda o=outer,k=kind,i=indices:backend.execute(o,k,i))
        if hasattr(backend,'write_histories') and kind in ('baseline','data','SDF','post_sdf_validation','cross_block','outer_commit'):
            backend.write_histories(journal.results,figures=kind in ('baseline','outer_commit'))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sampling-dir','initial-checkpoint','source-manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--parent-batch-size',type=int,default=128)
    p.add_argument('--fit-parent-batch-size',type=int,default=128)
    p.add_argument('--firm-batch-size',type=int,default=512)
    p.add_argument('--approval',type=Path)
    p.add_argument('--execute-approved',action='store_true')
    p.add_argument('--resume',action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    args=parse_args(argv); plan=training_plan()
    if not args.execute_approved:
        print(json.dumps(dict(CURRENT_PHASE='WAITING_FOR_USER_REVIEW',FORMAL_TRAINING_STARTED=False,
            SLURM_JOB_SUBMITTED=False,plan=plan,plan_fingerprint=stable_json_hash(plan)),indent=2));return
    if args.approval is None: raise ValueError('explicit continuation approval required')
    approved=json.loads(args.approval.read_text())
    if approved.get('approved') is not True or approved.get('schema_version')!=4:
        raise ValueError('explicit dynamic v4 approval required; old v3 forbidden')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    expected=approval_binding(plan,commit,args.sampling_dir,args.initial_checkpoint,args.source_manifest)
    require_continuation_approval(approved,expected)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('clean approved commit required')
    if torch.device(args.device).type!='cuda' or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('formal continuation requires exactly one visible CUDA GPU')
    if 'A800' not in torch.cuda.get_device_name(torch.device(args.device)):
        raise ValueError('reviewed formal hardware is one A800')
    if min(args.parent_batch_size,args.fit_parent_batch_size,args.firm_batch_size)<=0: raise ValueError('invalid physical batch')
    locked=load_locked_sampling(args.sampling_dir)
    if locked[2].get('run_identity')!=RUN_ID or locked[2]['config']['train_parents']!=8192 or locked[2]['config']['validation_parents']!=2048 or locked[2]['config']['firms']!=200:
        raise ValueError('wrong continuation sampling identity/size')
    for split,n in (('train',8192),('validation',2048)):
        rows=locked[1][split]['parents']
        if len(rows)!=n or locked[0][split]['state']['b'].shape!=(n,200): raise ValueError('incomplete logical bank')
        if any(sum(r['sampling_source']==kind for r in rows)!=n//2 for kind in ('BETA','EMPIRICAL')):
            raise ValueError('wrong Parent mixture ratio')
    initial,source=verify_initial(args.initial_checkpoint,args.source_manifest,locked[2]['fingerprint'])
    if source.get('new_code_commit')!=commit: raise ValueError('prepared manifest must bind this reviewed clean commit')
    pointer=args.output/'transactions/latest.json'
    if args.resume:
        if not pointer.exists(): raise ValueError('INTRA_RUN_RESUME requires this new run committed pointer')
    elif args.output.exists() and list(args.output.iterdir()): raise ValueError('fresh run must be empty')
    backend=ContinuationBackend(args,locked,initial,source)
    contract=dict(source_commit=commit,plan=plan,approval_binding=expected,run_identity=plan['experiment'],
        sampling_signature=locked[2],economic=asdict(backend.economic),hyperparameters=asdict(backend.episode.hyperparams),
        runtime=dict(device=args.device,parent_batch=args.parent_batch_size,fit_batch=args.fit_parent_batch_size,firm_batch=args.firm_batch_size))
    journal=StageJournal(args.output/'transactions',contract,backend,resume=args.resume)
    def history_init():
        result=install_historical_baseline(args.initial_checkpoint.parent,args.output,source)
        atomic_write_bytes(args.output/'metadata/continuation_source_manifest.json',args.source_manifest.read_bytes())
        atomic_write_json(args.output/'metadata/continuation_training_plan.json',plan)
        atomic_write_bytes(args.output/'metadata/approval_v3.json',args.approval.read_bytes())
        return result
    journal.run('initialization/historical_baseline',history_init)
    install_historical_baseline(args.initial_checkpoint.parent,args.output,source,resume=True)
    drive(backend,journal)
    atomic_write_json(args.output/'final_summary.json',dict(ADDITIONAL_OUTERS_COMPLETED=8,
        OUTER_START=4,OUTER_END=11,source_commit=commit,source_job_id=701387,
        ECONOMIC_EQUILIBRIUM_CONVERGED='NOT_ASSERTED',completed=journal.completed,
        final_hashes=component_hashes({k:m.state_dict() for k,m in backend.models.items()})))


if __name__=='__main__':main()
