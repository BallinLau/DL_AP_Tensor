"""Review-gated H50 adapter, defaults to plan-only; never creates approval."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import torch
from analysis.finite_direct_continuation_state import safe_load,verify_optimizer_restored
from analysis.finite_direct_outer4_state import StageJournal
from analysis.nested_equilibrium_utils import (stable_json_hash,atomic_write_json,atomic_torch_save,
    restore_training_transaction,module_hash)
from analysis.run_full_nested_finite_direct_outer4 import (ProductionBackend,Budget,stages,
    stage_id,restore_fresh,set_scope)
from analysis.run_fc1_self_consistent_macro import hashes
from data.finite_direct_initial_hstar import VERSION
from data.finite_direct_long_horizon_runtime import generate_long_horizon_dataset,ShardStore
from data.finite_direct_long_horizon_sampling import broad_firm_bank,sample_macro_rows
from data.finite_direct_runtime import load_locked_sampling,check_firm_lineage
from data.finite_direct_sampling import file_hash
from data.market_clearing import MarketClearingConfig
from analysis.h50_source_certification import CONTRACT, certify, save_bank_reference, load_bank_reference, verify_fixed_validation
from data.finite_direct_joint_coverage import JointCoverageConfig,build_joint_bank

SCHEMA='CONTINUATION_APPROVAL_V5_HSTAR_H50'


def coverage_config(args=None):
    path=getattr(args,'coverage_config',None)
    cfg=JointCoverageConfig(**json.loads(path.read_text())) if path else JointCoverageConfig()
    cfg.validate();return cfg


def plan(coverage=None):
    # Epoch sample budgets are proposed for explicit approval, not hidden 50x
    # optimizer expansion. All available time bins are eligible in sampling.
    execution=[dict(outer=12,kind='baseline',indices={})]
    for o,k,i in stages(replace(Budget(),equilibrium_outer_max=5)):
        if o<1:continue
        if k=='cross_block':execution.append(dict(outer=o+11,kind='EMPIRICAL_POOL_COMMIT',indices={}))
        execution.append(dict(outer=o+11,kind=k,indices=i))
    return dict(version=VERSION,continuation_contract=CONTRACT,horizon=50,primary_branch=0,children=2,
        initial_train_rule='CERTIFIED_OUTER11_BASE; NEW_IMMUTABLE_JOINT_COVERAGE_TRAIN_BANK',
        joint_coverage=asdict(coverage or JointCoverageConfig()),
        start_outer=12,end_outer=15,train_parents=8192,validation_parents=2048,firms=200,
        initial_root=asdict(MarketClearingConfig()),root_only_t0=True,
        finite_operator='finite_direct_parent_exit_before_child_v2',
        budget=asdict(Budget()),firm_rows=dict(train=8192,validation=2048),
        calculated_fraction=.5,macro_pair_budget=dict(train=8192,validation=2048),
        sdf_pair_budget=dict(train=8192,validation=2048),
        optimizer_budget='UNCHANGED; BOUNDED_LOCKED_SUBSET_PER_DATA_STAGE',
        firm_sampling='LINEAGE_TIME_MACRO_STRATIFIED',time_strata=[[0,9],[10,29],[30,49]],
        beta_empirical='TARGET_50:50; CERTIFIED_EMPIRICAL_SHORTFALL_FILLED_BY_ROOT_VALID_BETA',recent_archive='70:30_WHEN_AVAILABLE',
        validation='FIXED_INITIAL_BANK_AND_CRN; HSTAR_RECALIBRATED_IF_MODEL_CHANGES',
        training_distribution_is_not_population=True,fc2=False,
        stages=execution)


def approval_binding(args,commit):
    sampling=args.sampling_dir
    required=['sampling_signature.json','train_initial_bank.pt','validation_initial_bank.pt']
    return dict(schema_name=SCHEMA,schema_version=5,source_commit=commit,
        plan_fingerprint=stable_json_hash(plan(coverage_config(args))),
        joint_coverage_fingerprint=stable_json_hash(asdict(coverage_config(args))),
        sampling_signature_fingerprint=json.loads((sampling/'sampling_signature.json').read_text())['fingerprint'],
        source_checkpoint_sha256=file_hash(args.initial_checkpoint),
        committed_source_journal_sha256=file_hash(args.source_journal_checkpoint),
        committed_source_pointer_sha256=file_hash(args.source_latest_pointer),
        source_job=702636,continuation_contract=CONTRACT,
        dynamic_source_artifact_sha256={k:v['sha256'] for k,v in
            safe_load(args.initial_checkpoint)['state']['dynamic_banks']['11'].items()
            if k in ('pool','bank','schedule')},
        input_artifact_sha256={n:file_hash(sampling/n) for n in required},
        root_fingerprint=stable_json_hash(plan()['initial_root']),
        horizon=50,operator=VERSION)


def require_approval(args,commit):
    if not args.execute_approved:return False
    if args.approval is None:raise ValueError('explicit V5 approval required')
    supplied=json.loads(args.approval.read_text())
    if supplied.get('schema_version')!=5 or supplied.get('schema_name')!=SCHEMA:
        raise ValueError('V5 content-bound approval mismatch; v4 forbidden')
    if any(getattr(args,k,None) is None for k in ('sampling_dir','initial_checkpoint','source_journal_checkpoint','source_latest_pointer')):
        raise ValueError('V5 certified source artifact paths required')
    expected=approval_binding(args,commit)
    if supplied.get('approved') is not True or any(supplied.get(k)!=v for k,v in expected.items()):
        raise ValueError('V5 content-bound approval mismatch; v4 forbidden')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('clean source checkout required')
    return True


class H50Backend(ProductionBackend):
    """Calls original P/Q/BP/FC1/SDF optimizers only after V5 approval.

    No optimization is executed by this review task. The bounded macro subset
    stays on CPU until existing minibatches use it. Complete rollout shards
    remain available for population/coverage diagnostics.
    """
    def __init__(self,args,locked,learning):
        args.initial_hstar_h50_adapter=True
        super().__init__(args,locked,learning_snapshot=learning)
        restore_training_transaction(learning['transaction'],models=self.models,
            optimizers={**self.episode.optimizers,'fc1_conditional':self.fc1},episode=self.episode)
        restore_fresh(self.episode,learning['fresh'])
        self.state['sdf_epoch']=learning['state']['sdf_epoch']
        self.episode.refresh_firm_target()
        self.state['initialization']='FULL_OUTER11_LEARNING_STATE_NEW_H50_DATA_CONTRACT'
        self.state['pool_sources']=[]
        _,bank,schedule,pool,audits=certify(args.initial_checkpoint,args.source_journal_checkpoint,
            args.source_latest_pointer,source_job=702636,artifact_dir=args.source_artifact_dir)
        verify_fixed_validation(args.sampling_dir)
        if set(bank['state']['transition_path_id'].tolist()) & set(self.banks['validation']['state']['transition_path_id'].tolist()):
            raise ValueError('source dynamic Train/Validation lineage overlap')
        self.banks['train']=bank;self.schedules['train']=schedule
        self.initial_train=bank;self.initial_schedule=schedule
        self.archive_templates=pool['archive']+pool['recent']
        self.state['h50_active_outer']=12
        self.state['source_certification']=audits
        self.state['restored_optimizer_audit']={name:verify_optimizer_restored(
            learning['transaction']['optimizers'][name],opt) for name,opt in
            {**self.episode.optimizers,'fc1_conditional':self.fc1}.items()}
        self._source_bank_ref=save_bank_reference(self.root,bank,schedule,outer=11,pool_version=pool['fingerprint'])
        self._active_bank_ref=save_bank_reference(self.root,bank,schedule,outer=12,pool_version=pool['fingerprint'])
        self.coverage_cfg=coverage_config(args)
        for name,audit in audits.items():atomic_write_json(self.root/'metadata'/f'{name}.json',audit)

    def _store(self,outer,stage,split,capture=True,templates=False):
        set_scope(self.models,'frozen')
        if module_hash(self.episode._target_policy_value())!=module_hash(self.models['policy_value']):
            raise ValueError('stale Firm Grid target')
        path=self.root/f'outer_{outer:03d}'/stage/split
        store=generate_long_horizon_dataset(self.models,self.episode,self.banks[split],
            self.schedules[split],self.signature,path,outer=outer,stage=stage,
            parent_batch_size=self.args.parent_batch_size,capture=capture,
            predecision_templates=templates,diagnostic_metadata=dict(
                initial_bank_sha=(self._active_bank_ref['sha256'] if split=='train' else file_hash(self.args.sampling_dir/'validation_initial_bank.pt')),
                coverage_config=asdict(self.coverage_cfg)))
        from analysis.h50_joint_diagnostics import write_dashboard,compare_validation_stages
        write_dashboard(store,path/'diagnostics',self.coverage_cfg)
        if split=='validation':compare_validation_stages({name:self.root/f'outer_{outer:03d}'/name/'validation/diagnostics' for name in ('D0','D1','D2')},self.root/f'outer_{outer:03d}'/'diagnostics')
        return store

    def _joint_train_bank(self,outer):
        from analysis.run_fc1_self_consistent_macro import simulator
        from analysis.economic_config import AnalysisEconomicConfig
        from data.finite_direct_sampling import state_hash
        def templates():
            for source in reversed(self.state['pool_sources']):
                path=Path(source['path'])
                if source['outer']>=outer or file_hash(path/'manifest.json')!=source['manifest_sha256']:raise ValueError('uncommitted/future empirical source')
                for shard in ShardStore(path).chunks():
                    for item in shard['templates']:
                        if 'dynamic_template' in item:yield item['dynamic_template']
            yield from self.archive_templates
        bank,schedule,stats=build_joint_bank(self.initial_train,templates(),outer=outer,
            target=plan()['train_parents'],cfg=self.coverage_cfg,output=self.root/f'outer_{outer:03d}'/'coverage_bank',
            simulator_factory=lambda n:simulator(self.models,self.episode,n,self.coverage_cfg.seed),
            economic=dict(config=asdict(AnalysisEconomicConfig.from_current_config()),hyperparameters=asdict(self.episode.hyperparams)),
            committed=True,pool_fingerprint=stable_json_hash(dict(
                source_sha256=self._source_bank_ref['sha256'],
                pools=[dict(outer=p['outer'],manifest_sha256=p['manifest_sha256']) for p in self.state['pool_sources']])) )
        self.banks['train']=bank;self.schedules['train']=schedule
        self.state.setdefault('coverage_banks',{})[str(outer)]=stats
        return stats

    def _generate(self,outer,stage,*,capture=True):
        result={}
        for split in ('train','validation'):
            store=self._store(outer,stage,split,capture)
            frame,index=sample_macro_rows(store,pair_budget=plan()['macro_pair_budget'][split],
                seed=20261021+(0 if split=='train' else 100000),
                source_by_id={r['parent_id']:r for r in self.schedules[split]['parents']} if split=='train' else None,
                coverage_config=self.coverage_cfg)
            frame.to_csv(store.root/'observations.csv',index=False)
            atomic_write_json(store.root/'sample_index.json',index)
            manifest=dict(store.manifest,observations_sha256=file_hash(store.root/'observations.csv'),
                          sampled_rows=len(frame),sampling=index)
            if stage=='D2':
                sdf_rows,sdf_index=sample_macro_rows(store,pair_budget=plan()['sdf_pair_budget'][split],
                    seed=20261022+(0 if split=='train' else 100000),complete_pairs_only=True,
                    source_by_id={r['parent_id']:r for r in self.schedules[split]['parents']} if split=='train' else None,
                    coverage_config=self.coverage_cfg)
                sdf_rows.to_csv(store.root/'sdf_observations.csv',index=False)
                atomic_write_json(store.root/'sdf_sample_index.json',sdf_index)
                manifest.update(sdf_observations_sha256=file_hash(store.root/'sdf_observations.csv'),sdf_sampling=sdf_index)
            atomic_write_json(store.root/'manifest.json',manifest)
            # _store already collected the canonical diagnostics once. Do not
            # reread every H50 shard merely to create a second figures folder.
            result[split]=(frame,[],manifest,store.root)
        return result

    def _data(self,outer,version,split):
        if not getattr(self,'_sdf_sampling',False):return super()._data(outer,version,split)
        record=self.state['data'][f'{outer}/D{version}/{split}'];path=Path(record['path'])
        if file_hash(path/'manifest.json')!=record['manifest_sha256']:raise ValueError('SDF H50 manifest corruption')
        manifest=json.loads((path/'manifest.json').read_text())
        if file_hash(path/'sdf_observations.csv')!=manifest['sdf_observations_sha256']:
            raise ValueError('SDF H50 subset corruption')
        return pd.read_csv(path/'sdf_observations.csv')

    def _sdf_chunk(self,outer,chunk):
        self._sdf_sampling=True
        try:return super()._sdf_chunk(outer,chunk)
        finally:self._sdf_sampling=False

    def _execute(self,outer,kind,i):
        directory=self.root/f'outer_{outer:03d}';directory.mkdir(parents=True,exist_ok=True)
        if kind=='baseline':
            from analysis.h50_diagnostics import streaming_economic_metrics
            store=self._store(outer,'firm_source','validation')
            bank=broad_firm_bank(store,split='validation',max_rows=2048,seed=20361021,
                calculated_fraction=plan()['calculated_fraction'])
            children=self._conditional(bank,bank.seed+1)
            path=self.root/'immutable_artifacts/fixed_firm_probe.pt'
            atomic_torch_save(path,dict(banks={'validation':bank.payload()},
                children={'validation':children.payload()}))
            self.state['probe_bank']=dict(path=str(path),sha256=file_hash(path))
            return dict(optimizer_steps=0,evaluation='NEW_H50_OPERATOR_BASELINE_BEFORE_FIRST_FIRM_UPDATE',
                economic_metrics=streaming_economic_metrics(store,self.schedules['validation'],outer=outer,stage='BASELINE'),
                fixed_probe_residual=self._cross(outer,label='initial_baseline'))
        if kind=='outer_start':
            if outer>12:
                expected=self.state['outer_metrics'][str(outer-1)]['committed_hashes']
                if hashes(self.models)!=expected:raise ValueError('previous committed model mismatch')
            coverage_stats=self._joint_train_bank(outer) if self.coverage_cfg.enabled else None
            self.state['h50_active_outer']=outer
            self._active_bank_ref=save_bank_reference(self.root,self.banks['train'],self.schedules['train'],
                outer=outer,pool_version=self.banks['train']['empirical_pool_fingerprint'])
            self.state['outer_metrics'][str(outer)]={'initial_hashes':hashes(self.models)}
            return dict(version=VERSION,source_outer=11,optimizer_steps=0,joint_coverage=coverage_stats)
        if kind=='firm_bank':
            set_scope(self.models,'frozen');banks={};children={}
            for split in ('train','validation'):
                store=self._store(outer,'firm_source',split)
                banks[split]=broad_firm_bank(store,split=split,max_rows=plan()['firm_rows'][split],
                    seed=20261021+(0 if split=='train' else 100000),
                    calculated_fraction=plan()['calculated_fraction'])
                # Never reuse stored RHS or old forecast-state targets.
                children[split]=self._conditional(banks[split],banks[split].seed+1)
            check_firm_lineage(banks['train'],banks['validation'])
            path=directory/'firm_conditional_bank.pt'
            payload=dict(banks={s:b.payload() for s,b in banks.items()},
                children={s:b.payload() for s,b in children.items()},
                macro_hash=module_hash(self.models['sdf_fc1']))
            atomic_torch_save(path,payload)
            self.state['firm_banks'][str(outer)]=dict(path=str(path),sha256=file_hash(path))
            if 'probe_bank' not in self.state:self.state['probe_bank']=dict(path=str(path),sha256=file_hash(path))
            return dict(optimizer_steps=0,source='MULTI_TIME_MACRO_STATES',
                max_source_time={s:int(b.metadata['source_time'].max()) for s,b in banks.items()},
                macro_hash=payload['macro_hash'])
        if kind=='data':
            from analysis.h50_diagnostics import streaming_economic_metrics
            from analysis.run_fc1_self_consistent_macro import evaluate
            version=i['version'];generated=self._generate(outer,f'D{version}')
            result={'economic_metrics':[]}
            for split,(frame,_,manifest,path) in generated.items():
                self.state['data'][f'{outer}/D{version}/{split}']=dict(path=str(path),manifest_sha256=file_hash(path/'manifest.json'))
                result[split]=dict(optimizer_subset_rows=len(frame),evaluation_scope='BOUNDED_TIME_STRATIFIED_SUBSET',
                    metrics=evaluate(self.models['sdf_fc1'],frame,self.episode.device),
                    scheduled_parent_times=manifest['scheduled_parent_times'])
                result['economic_metrics'].extend(streaming_economic_metrics(ShardStore(path),self.schedules[split],
                    outer=outer,stage=f'D{version}'))
            if version==0:result['pre_macro_firm_residual']=self._cross(outer,label='pre_macro')
            return result
        if kind=='post_sdf_validation':
            # Dedicated current-model provenance; never old D2.
            store=self._store(outer,'POST_SDF_VALIDATION_RESIMULATION','validation',capture=False)
            from analysis.h50_diagnostics import streaming_economic_metrics
            return dict(optimizer_steps=0,manifest=str(store.root/'manifest.json'),
                        evaluation='POST_SDF_VALIDATION_RESIMULATION',
                        economic_metrics=streaming_economic_metrics(store,self.schedules['validation'],
                            outer=outer,stage='POST_SDF_VALIDATION_RESIMULATION'))
        if kind=='EMPIRICAL_POOL_COMMIT':
            store=self._store(outer,'POST_SDF_TRAIN_EMPIRICAL_SOURCE','train',capture=False,templates=True)
            record=dict(path=str(store.root),manifest_sha256=file_hash(store.root/'manifest.json'),outer=outer)
            self.state['pool_sources'].append(record)
            atomic_write_json(directory/'dynamic_pool_commit.json',dict(sources=self.state['pool_sources']))
            return dict(optimizer_steps=0,committed_source=record)
        return super()._execute(outer,kind,i)

    def snapshot(self):
        result=super().snapshot()
        result['h50_train_bank_reference']=self._active_bank_ref
        result['h50_source_bank_reference']=self._source_bank_ref
        return result

    def restore(self,snapshot):
        super().restore(snapshot)
        if 'h50_train_bank_reference' not in snapshot:
            raise ValueError('old inline H50 snapshot requires explicit migration, not silent resume')
        ref=snapshot['h50_train_bank_reference']
        if ref['outer']!=self.state['h50_active_outer']:raise ValueError('active Outer/Bank mismatch')
        self.banks['train'],self.schedules['train']=load_bank_reference(ref)
        self.initial_train,self.initial_schedule=load_bank_reference(snapshot['h50_source_bank_reference'])
        self._active_bank_ref=ref;self._source_bank_ref=snapshot['h50_source_bank_reference']
        for record in self.state['data'].values():
            path=Path(record['path'])
            if file_hash(path/'manifest.json')!=record['manifest_sha256']:
                raise ValueError('committed H50 data manifest corruption on resume')
            store=ShardStore(path)
            if file_hash(path/'observations.csv')!=store.manifest['observations_sha256']:
                raise ValueError('committed H50 sample corruption on resume')
            if 'sdf_observations_sha256' in store.manifest and file_hash(path/'sdf_observations.csv')!=store.manifest['sdf_observations_sha256']:
                raise ValueError('committed SDF H50 sample corruption on resume')
            for name,digest in store.manifest['shards'].items():
                if Path(name).name!=name or file_hash(path/name)!=digest:
                    raise ValueError('committed H50 shard corruption on resume')
        for record in self.state['firm_banks'].values():
            if file_hash(record['path'])!=record['sha256']:
                raise ValueError('committed H50 Firm bank corruption on resume')
        for record in self.state['pool_sources']:
            path=Path(record['path'])
            if file_hash(path/'manifest.json')!=record['manifest_sha256']:
                raise ValueError('committed H50 pool corruption on resume')
            for name,digest in ShardStore(path).manifest['shards'].items():
                if Path(name).name!=name or file_hash(path/name)!=digest:
                    raise ValueError('committed H50 pool shard corruption on resume')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sampling-dir',type=Path)
    p.add_argument('--initial-checkpoint',type=Path)
    p.add_argument('--source-journal-checkpoint',type=Path)
    p.add_argument('--source-latest-pointer',type=Path)
    p.add_argument('--source-artifact-dir',type=Path)
    p.add_argument('--coverage-config',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--approval',type=Path)
    p.add_argument('--execute-approved',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--parent-batch-size',type=int,default=128)
    p.add_argument('--fit-parent-batch-size',type=int,default=128)
    p.add_argument('--firm-batch-size',type=int,default=512)
    args=p.parse_args(argv)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if not require_approval(args,commit):
        args.output.mkdir(parents=True,exist_ok=True)
        atomic_write_json(args.output/'training_plan.json',plan(coverage_config(args)))
        return 0
    if args.sampling_dir is None or args.initial_checkpoint is None:
        raise ValueError('approved artifact paths required')
    locked=load_locked_sampling(args.sampling_dir)
    learning=safe_load(args.initial_checkpoint)
    if not {'transaction','fresh','state'} <= set(learning):
        raise ValueError('full committed learning state required, not weights-only')
    backend=H50Backend(args,locked,learning)
    contract=approval_binding(args,commit)
    journal=StageJournal(args.output/'transactions',contract,backend,resume=args.resume)
    execution=plan()['stages']
    for stage in execution:
        o,k,i=stage['outer'],stage['kind'],stage['indices']
        journal.run(stage_id(o,k,i),lambda o=o,k=k,i=i:backend.execute(o,k,i))
        if k in ('data','SDF','cross_block','outer_commit'):
            backend.write_histories(journal.results,figures=k=='outer_commit')
    return 0


if __name__=='__main__':raise SystemExit(main())
