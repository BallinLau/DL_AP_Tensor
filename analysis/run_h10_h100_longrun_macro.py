"""Independent H10 Outer / historical H100 FC1 replay; explicit V6 approval."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import time

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
from data.finite_direct_replay_runtime import generate_long_horizon_dataset,ShardStore
from data.finite_direct_longrun_replay import load_pool,select_mixture,fit_edges,commit_pool
from data.finite_direct_long_horizon_sampling import broad_firm_bank as _broad_firm_bank
from data.finite_direct_runtime import load_locked_sampling,check_firm_lineage
from data.finite_direct_sampling import file_hash
from data.market_clearing import MarketClearingConfig
from analysis.h50_source_certification import CONTRACT, certify, save_bank_reference, load_bank_reference, verify_fixed_validation
from data.finite_direct_joint_coverage import JointCoverageConfig,build_joint_bank
from data.finite_direct_uniform_firm_macro import FirmMacroCoverageConfig,configured_macro_bank,firm_coverage_summary

SCHEMA='CONTINUATION_APPROVAL_V6_H10_H100'


def broad_firm_bank(*args,**kwargs):
    return _broad_firm_bank(*args,**kwargs,time_boundaries=(3,7))


def distinct_balanced_donors(bank):
    """Do not label repeated capture draws independent; keep bounded real rows.

    No new micro or macro state is fabricated. Underfilled observed samples
    retain balanced ON_POLICY/CALCULATED support before explicit Hybrid split.
    """
    from copy import deepcopy
    from training.inner_fixed_point import InnerFPCollocationBank
    meta=bank.metadata;n=len(bank.parent_states);seen=set();groups=[[],[]]
    for j,key in enumerate(zip(*(meta[k].tolist() for k in ('path','source_time','source_branch','firm_id')))):
        if key in seen:continue
        seen.add(key);groups[int('CALCULATED' in meta['source_kind'][j])].append(j)
    count=min(map(len,groups));ids=torch.tensor(groups[0][:count]+groups[1][:count],dtype=torch.long)
    if not len(ids):raise ValueError('NO_DISTINCT_BALANCED_FIRM_DONORS')
    result=deepcopy(meta)
    for k,v in list(result.items()):
        if torch.is_tensor(v) and v.ndim and len(v)==n:result[k]=v[ids].clone()
        elif isinstance(v,list) and len(v)==n:result[k]=[v[j] for j in ids.tolist()]
    result.update(requested_rows=n,actual_distinct_donor_rows=len(ids),donor_shortfall=n-len(ids),
        unique_source_firm_time_count=len(ids),actual_draws=len(ids))
    return InnerFPCollocationBank(bank.parent_states[ids].clone(),bank.source_kind[ids].clone(),bank.seed,bank.split,result)


def coverage_config(args=None):
    path=getattr(args,'coverage_config',None)
    cfg=JointCoverageConfig(**json.loads(path.read_text())) if path else JointCoverageConfig()
    cfg.validate();return cfg


def firm_coverage_config(args=None):
    path=getattr(args,'firm_macro_config',None)
    cfg=FirmMacroCoverageConfig(**json.loads(path.read_text())) if path else FirmMacroCoverageConfig(
        firm_macro_coverage_enabled=True,firm_macro_coverage_fraction=.5)
    cfg.validate();return cfg


def plan(coverage=None,firm_coverage=None,*,end_outer=15,skip_bp_head=False):
    # Epoch sample budgets are proposed for explicit approval, not hidden 50x
    # optimizer expansion. All available time bins are eligible in sampling.
    firm_coverage=firm_coverage or firm_coverage_config()
    effective_firm_rows=(dict(train=firm_coverage.firm_macro_sampling_budget,validation=min(2048,firm_coverage.firm_macro_sampling_budget))
        if firm_coverage.firm_macro_coverage_enabled else dict(train=8192,validation=2048))
    execution=[dict(outer=12,kind='baseline',indices={})]
    if end_outer!=15:raise ValueError('new experiment budget is exactly Outer12..15')
    for o,k,i in stages(replace(Budget(),equilibrium_outer_max=end_outer-10)):
        if o<1:continue
        if k=='cross_block':execution.append(dict(outer=o+11,kind='EMPIRICAL_POOL_COMMIT',indices={}))
        execution.append(dict(outer=o+11,kind=k,indices=i))
        if k=='outer_commit':execution.append(dict(outer=o+11,kind='LONGRUN_MACRO_POOL_COMMIT',indices={}))
    return dict(version='H10_H100_LONGRUN_V1',continuation_contract=CONTRACT,horizon=10,primary_branch=0,children=2,
        longrun=dict(paths=200,firms=200,horizon=100,branches=1,burnin=20,
            root_consistent=100,wide_belief=100,initial_design='SCRAMBLED_SOBOL',
            semantics='HISTORICAL_POLICY_MACRO_REPLAY',no_optimizer=True),
        initial_train_rule='CERTIFIED_OUTER11_BASE; NEW_IMMUTABLE_JOINT_COVERAGE_TRAIN_BANK',
        joint_coverage=asdict(coverage or JointCoverageConfig()),
        firm_macro_coverage=asdict(firm_coverage),
        effective_firm_batch_size=firm_coverage.firm_macro_batch_size if firm_coverage.firm_macro_coverage_enabled else 512,
        start_outer=12,end_outer=end_outer,train_parents=4096,validation_parents=1024,firms=200,
        bp_head_training_skipped_by_user=skip_bp_head,simulation_bp_action_source='grid',
        firm_donor_sampling='DISTINCT_FIRM_TIME_BRANCH_BALANCED_OBSERVED_SUBSET',
        sdf_no_pairs='SKIP_WITHOUT_OPTIMIZER_OR_AL_MUTATION',
        initial_root=asdict(MarketClearingConfig()),root_only_t0=True,
        finite_operator='finite_direct_parent_exit_before_child_v2',
        budget=asdict(Budget()),firm_rows=effective_firm_rows,
        calculated_fraction=.5,macro_pair_budget=dict(train=4096,validation=10240),
        sdf_pair_budget=dict(train=4096,validation=10240),
        fc1_mixture=dict(longrun=2048,coverage=2048,coverage_shortfall_fill=True,lineage_cap_fraction=.05),
        optimizer_budget='UNCHANGED; BOUNDED_LOCKED_SUBSET_PER_DATA_STAGE',
        firm_sampling='LINEAGE_TIME_MACRO_STRATIFIED',time_strata=[[0,2],[3,6],[7,9]],
        beta_empirical='TARGET_50:50; CERTIFIED_EMPIRICAL_SHORTFALL_FILLED_BY_ROOT_VALID_BETA',recent_archive='70:30_WHEN_AVAILABLE',
        validation='FIXED_INITIAL_BANK_AND_CRN; HSTAR_RECALIBRATED_IF_MODEL_CHANGES',
        training_distribution_is_not_population=True,fc2=False,
        stages=execution)


def approval_binding(args,commit):
    sampling=args.sampling_dir
    required=['sampling_signature.json','train_initial_bank.pt','validation_initial_bank.pt']
    return dict(schema_name=SCHEMA,schema_version=6,source_commit=commit,
        plan_fingerprint=stable_json_hash(plan(coverage_config(args),firm_coverage_config(args),
            end_outer=getattr(args,'end_outer',15),skip_bp_head=getattr(args,'skip_bp_head',False))),
        runtime_partition=dict(parent_batch_size=getattr(args,'parent_batch_size',128),
            fit_parent_batch_size=getattr(args,'fit_parent_batch_size',128),firm_batch_size=getattr(args,'firm_batch_size',512)),
        firm_macro_coverage_fingerprint=stable_json_hash(asdict(firm_coverage_config(args))),
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
        horizon=10,operator=VERSION)


def require_approval(args,commit):
    if not args.execute_approved:return False
    if args.approval is None:raise ValueError('explicit V6 approval required')
    supplied=json.loads(args.approval.read_text())
    if supplied.get('schema_version')!=6 or supplied.get('schema_name')!=SCHEMA:
        raise ValueError('V6 content-bound approval mismatch; H50 V5 forbidden')
    if any(getattr(args,k,None) is None for k in ('sampling_dir','initial_checkpoint','source_journal_checkpoint','source_latest_pointer')):
        raise ValueError('V5 certified source artifact paths required')
    expected=approval_binding(args,commit)
    if supplied.get('approved') is not True or any(supplied.get(k)!=v for k,v in expected.items()):
        raise ValueError('V6 content-bound approval mismatch; H50 V5 forbidden')
    protection=json.loads((ROOT/'H50_PROTECTION_MANIFEST.json').read_text())
    for protected in (Path(protection['source_worktree']).resolve(),Path(protection['output_directory']).resolve()):
        for candidate in (ROOT.resolve(),args.output.resolve()):
            if candidate==protected or candidate.is_relative_to(protected) or protected.is_relative_to(candidate):
                raise ValueError('H50 protected source/output overlap')
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
        # Keep fixed original Validation identities/CRN, never resample by fit.
        from data.finite_direct_sampling import state_hash
        from analysis.run_fc1_self_consistent_macro import select_state
        for split,target in [('validation',getattr(args,'smoke_parents',1024))]:
            original=self.banks[split];indices=torch.arange(target)
            if len(original['state']['x'])<target:raise ValueError('insufficient fixed Validation bank')
            original=dict(original,state=select_state(original['state'],indices,torch.device('cpu')))
            original['fingerprint']=state_hash(original['state'])
            schedule=dict(self.schedules[split],parents=self.schedules[split]['parents'][:target])
            schedule['fingerprint']=stable_json_hash(schedule['parents'])
            original['schedule_fingerprint']=schedule['fingerprint']
            self.banks[split]=original;self.schedules[split]=schedule
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
        self.firm_coverage_cfg=firm_coverage_config(args)
        for name,audit in audits.items():atomic_write_json(self.root/'metadata'/f'{name}.json',audit)
        if self.firm_coverage_cfg.firm_macro_coverage_enabled:
            if self.firm_coverage_cfg.firm_macro_coverage_mode!='HYBRID':raise ValueError('STANDALONE uses reviewed Firm-only runner, not Outer')
            args.firm_batch_size=self.firm_coverage_cfg.firm_macro_batch_size

    def _conditional(self,bank,seed):
        from data.finite_direct_runtime import conditional_child_bank
        return conditional_child_bank(self.models['sdf_fc1'],bank,seed=seed,n_child_shocks=2,
            economic=self.economic,device=self.episode.device,chunk_size=self.args.firm_batch_size,
            m_min=self.episode.hyperparams.pv_m_clamp_min,m_max=self.episode.hyperparams.pv_m_clamp_max)

    def _firm_coverage_bank(self,bank,outer):
        cfg=self.firm_coverage_cfg
        if getattr(self.args,'skip_bp_head',False):
            bank=distinct_balanced_donors(bank)
        return configured_macro_bank(bank,cfg,seed=cfg.firm_macro_sampling_seed+outer*100003+(100000 if bank.split=='validation' else 0))

    def _store(self,outer,stage,split,capture=True,templates=False):
        set_scope(self.models,'frozen')
        if module_hash(self.episode._target_policy_value())!=module_hash(self.models['policy_value']):
            raise ValueError('stale Firm Grid target')
        path=self.root/f'outer_{outer:03d}'/stage/split
        store=generate_long_horizon_dataset(self.models,self.episode,self.banks[split],
            self.schedules[split],self.signature,path,outer=outer,stage=stage,
            horizon=10,branches=2,
            parent_batch_size=self.args.parent_batch_size,capture=capture,
            predecision_templates=templates,diagnostic_metadata=dict(
                initial_bank_sha=(self._active_bank_ref['sha256'] if split=='train' else file_hash(self.args.sampling_dir/'validation_initial_bank.pt')),
                coverage_config=asdict(self.coverage_cfg)))
        # Heavy H50-specific plots/time bins are deliberately not on the
        # training critical path. Raw sidecars retain complete actual events.
        return store

    def _joint_train_bank(self,outer):
        from analysis.run_fc1_self_consistent_macro import simulator
        from analysis.economic_config import AnalysisEconomicConfig
        from data.finite_direct_sampling import state_hash
        def templates():
            from data.finite_direct_long_horizon_empirical import committed_templates
            yield from committed_templates(self.state['pool_sources'],outer=outer)
            yield from self.archive_templates
        bank,schedule,stats=build_joint_bank(self.initial_train,templates(),outer=outer,
            target=getattr(self.args,'smoke_parents',plan()['train_parents']),cfg=self.coverage_cfg,output=self.root/f'outer_{outer:03d}'/'coverage_bank',
            simulator_factory=lambda n:simulator(self.models,self.episode,n,self.coverage_cfg.seed),
            economic=dict(config=asdict(AnalysisEconomicConfig.from_current_config()),hyperparameters=asdict(self.episode.hyperparams)),
            committed=True,pool_fingerprint=stable_json_hash(dict(
                source_sha256=self._source_bank_ref['sha256'],
                pools=[dict(outer=p['outer'],manifest_sha256=p['manifest_sha256']) for p in self.state['pool_sources']])) )
        self.banks['train']=bank;self.schedules['train']=schedule
        self.state.setdefault('coverage_banks',{})[str(outer)]=stats
        return stats

    def _generate(self,outer,stage,*,capture=True):
        from data.finite_direct_longrun_replay import eligibility
        result={}
        for split in ('train','validation'):
            store=self._store(outer,stage,split,capture=False)
            # One columnar read/eligibility pass yields both training indices.
            pieces=[];pairs=[];attempted=0
            for chunk in store.rows():
                checked=eligibility(chunk);attempted+=len(chunk)
                pieces.append(checked.fc1);pairs.append(checked.sdf_pairs)
            frame=pd.concat(pieces,ignore_index=True)
            sdf_rows=pd.concat(pairs,ignore_index=True)
            index=dict(actual_unique_child_rows=len(frame),
                actual_parent_times=len(frame.drop_duplicates(['parent_candidate_id','t'])),
                attempted_rows=attempted,scheduled_rows=store.manifest['scheduled_child_rows'],
                rule='FULL_LEGAL_CURRENT_COVERAGE; MIXTURE_SELECTED_AT_FC1_STAGE',
                eligibility_passes=1)
            frame.to_csv(store.root/'observations.csv',index=False)
            atomic_write_json(store.root/'sample_index.json',index)
            manifest=dict(store.manifest,observations_sha256=file_hash(store.root/'observations.csv'),
                          sampled_rows=len(frame),sampling=index)
            if stage=='D2':
                # Bound groups, never split siblings. Stable index selection,
                # not a fabricated second Child from the historical replay.
                group=sdf_rows[['parent_candidate_id','t']].drop_duplicates()
                budget=plan()['sdf_pair_budget'][split]
                if len(group)>budget:
                    group=group.sample(n=budget,random_state=20261022)
                    sdf_rows=sdf_rows.merge(group,on=['parent_candidate_id','t'],how='inner',validate='many_to_one')
                sdf_index=dict(actual_parent_times=len(group),actual_unique_child_rows=len(sdf_rows),
                    complete_pairs_only=True,source='CURRENT_OUTER_D2_TWO_CHILD_ONLY',eligibility_passes=1)
                sdf_rows.to_csv(store.root/'sdf_observations.csv',index=False)
                atomic_write_json(store.root/'sdf_sample_index.json',sdf_index)
                manifest.update(sdf_observations_sha256=file_hash(store.root/'sdf_observations.csv'),sdf_sampling=sdf_index)
            atomic_write_json(store.root/'manifest.json',manifest)
            # _store already collected the canonical diagnostics once. Do not
            # reread every H50 shard merely to create a second figures folder.
            result[split]=(frame,[],manifest,store.root)
        return result

    def _data(self,outer,version,split):
        if split=='train' and getattr(self,'_fc1_replay_key',None):
            record=self.state['longrun_fc1_samples'][self._fc1_replay_key]
            if file_hash(record['path'])!=record['sha256']:raise ValueError('FC1 mixed dataset corruption')
            return pd.read_csv(record['path'])
        if not getattr(self,'_sdf_sampling',False):return super()._data(outer,version,split)
        record=self.state['data'][f'{outer}/D{version}/{split}'];path=Path(record['path'])
        if file_hash(path/'manifest.json')!=record['manifest_sha256']:raise ValueError('SDF H50 manifest corruption')
        manifest=json.loads((path/'manifest.json').read_text())
        if file_hash(path/'sdf_observations.csv')!=manifest['sdf_observations_sha256']:
            raise ValueError('SDF H50 subset corruption')
        return pd.read_csv(path/'sdf_observations.csv')

    def _fc1_epoch(self,outer,i):
        key=f'{outer}/{i["macro"]}'
        if key not in self.state.setdefault('longrun_fc1_samples',{}):
            coverage=super()._data(outer,i['macro'],'train')
            pool=self.state.get('longrun_pools',{}).get(str(outer-1))
            replay=load_pool(pool['path'],consumer_outer=outer) if pool else pd.DataFrame()
            edges=self.state.get('fc1_locked_edges')
            if edges is None:
                edges=fit_edges([coverage,replay]);self.state['fc1_locked_edges']=edges
            mixed,audit=select_mixture(coverage,replay,edges=edges,seed=20261022+outer*100+i['macro'])
            validation=super()._data(outer,i['macro'],'validation')
            if set(mixed.parent_candidate_id)&set(validation.parent_candidate_id):raise ValueError('FC1 replay Validation lineage leakage')
            path=self.root/f'outer_{outer:03d}'/f'FC1_{i["macro"]}_mixture.csv'
            mixed.to_csv(path,index=False)
            atomic_write_json(path.with_suffix('.json'),dict(audit=audit,edges=edges,semantics='HISTORICAL_POLICY_MACRO_REPLAY_PLUS_CURRENT_COVERAGE'))
            self.state['longrun_fc1_samples'][key]=dict(path=str(path),sha256=file_hash(path))
        # Existing production FC1 implementation calls self._data; scope the
        # input adapter, never replace its loss/optimizer or validation.
        self._fc1_replay_key=key
        try:return super()._fc1_epoch(outer,i)
        finally:self._fc1_replay_key=None

    def _sdf_chunk(self,outer,chunk):
        self._sdf_sampling=True
        try:
            frame=self._data(outer,2,'train')
            if frame.empty:
                return dict(status='SDF_SKIPPED_NO_VALID_PAIRS',optimizer_steps=0,epochs=0,
                    al_updated=False,learning_state_preserved=True)
            return super()._sdf_chunk(outer,chunk,allow_empty_validation=True)
        finally:self._sdf_sampling=False

    def _execute(self,outer,kind,i):
        directory=self.root/f'outer_{outer:03d}';directory.mkdir(parents=True,exist_ok=True)
        if kind=='BP' and getattr(self.args,'skip_bp_head',False):
            before=module_hash(self.models['policy_value'])
            if self.episode.hyperparams.simulation_bp_action_source!='grid':
                raise ValueError('Grid simulation mandatory; no BP head fallback')
            result=dict(BP_HEAD_TRAINING_SKIPPED_BY_USER=True,BP_HEAD_OPTIMIZER_STEPS=0,
                SIMULATION_BP_ACTION_SOURCE='GRID',optimizer_steps=0,optimizer_executed=False,
                model_hash_before=before,model_hash_after=module_hash(self.models['policy_value']))
            if result['model_hash_after']!=before:raise ValueError('skipped BP mutated Firm')
            return result
        if kind=='baseline':
            from analysis.h50_diagnostics import streaming_economic_metrics
            store=self._store(outer,'firm_source','validation')
            bank=broad_firm_bank(store,split='validation',max_rows=2048,seed=20361021,
                calculated_fraction=plan()['calculated_fraction'])
            bank=self._firm_coverage_bank(bank,outer)
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
                budget=(self.firm_coverage_cfg.firm_macro_sampling_budget if split=='train' else min(2048,self.firm_coverage_cfg.firm_macro_sampling_budget)) if self.firm_coverage_cfg.firm_macro_coverage_enabled else plan()['firm_rows'][split]
                banks[split]=broad_firm_bank(store,split=split,max_rows=budget,
                    seed=20261021+(0 if split=='train' else 100000),
                    calculated_fraction=plan()['calculated_fraction'])
                banks[split]=self._firm_coverage_bank(banks[split],outer)
                atomic_write_json(directory/f'firm_source_composition_{split}.json',firm_coverage_summary(banks[split]))
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
        if kind=='LONGRUN_MACRO_POOL_COMMIT':
            from data.finite_direct_h100_initial import design_initial
            checkpoint=directory/'outer_complete.pt'
            if not checkpoint.exists():raise ValueError('H100 requires committed Outer model')
            set_scope(self.models,'frozen');before=hashes(self.models)
            stores=[];design=[]
            for mode in ('ROOT_CONSISTENT','OFF_EQUILIBRIUM_INITIAL_BELIEF'):
                bank,schedule,meta=design_initial(self.initial_train,outer=outer,mode=mode)
                if hasattr(self.args,'smoke_longrun_paths'):
                    from analysis.run_fc1_self_consistent_macro import select_state
                    from data.finite_direct_sampling import state_hash
                    count=self.args.smoke_longrun_paths
                    bank=dict(bank,state=select_state(bank['state'],slice(0,count),torch.device('cpu')))
                    bank['fingerprint']=state_hash(bank['state'])
                    schedule=dict(schedule,parents=schedule['parents'][:count])
                    schedule['fingerprint']=stable_json_hash(schedule['parents'])
                    bank['schedule_fingerprint']=schedule['fingerprint']
                    meta=dict(meta,count=count,SMOKE_ONLY=True)
                design.append(meta)
                store=generate_long_horizon_dataset(self.models,self.episode,bank,schedule,self.signature,
                    directory/'longrun'/mode,outer=outer,stage='POST_OUTER_H100',horizon=100,branches=1,
                    capture=False,initial_mode=mode,parent_batch_size=self.args.parent_batch_size)
                stores.append(store)
            if hashes(self.models)!=before:raise ValueError('H100 mutated model')
            from analysis.nested_equilibrium_utils import module_hash
            model_hashes=dict(firm=before['firm'],fc1=module_hash(self.models['sdf_fc1'].fc1_model),sdf=before['sdf'])
            path=directory/'longrun_pool'
            result=commit_pool(stores,path,source_outer=outer,model_hashes=model_hashes,
                initial_design=design,outer_checkpoint_sha=file_hash(checkpoint))
            self.state.setdefault('longrun_pools',{})[str(outer)]=dict(path=str(path),fingerprint=result['fingerprint'])
            return dict(optimizer_steps=0,training_performed=False,pool=result)
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
        for key,record in self.state.get('longrun_fc1_samples',{}).items():
            if file_hash(record['path'])!=record['sha256']:raise ValueError('committed FC1 mixture corruption')
        for key,record in self.state.get('longrun_pools',{}).items():
            path=Path(record['path']);m=json.loads((path/'commit.json').read_text())
            if m['fingerprint']!=record['fingerprint']:raise ValueError('committed replay pointer mismatch')
            load_pool(path,consumer_outer=int(key)+1)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sampling-dir',type=Path)
    p.add_argument('--initial-checkpoint',type=Path)
    p.add_argument('--source-journal-checkpoint',type=Path)
    p.add_argument('--source-latest-pointer',type=Path)
    p.add_argument('--source-artifact-dir',type=Path)
    p.add_argument('--coverage-config',type=Path)
    p.add_argument('--firm-macro-config',type=Path)
    p.add_argument('--end-outer',type=int,default=15)
    p.add_argument('--skip-bp-head',action='store_true')
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
        atomic_write_json(args.output/'training_plan.json',plan(coverage_config(args),firm_coverage_config(args),end_outer=args.end_outer,skip_bp_head=args.skip_bp_head))
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
    atomic_write_json(args.output/'metadata'/'approval_v6.json',json.loads(args.approval.read_text()))
    execution=plan(coverage_config(args),firm_coverage_config(args),end_outer=args.end_outer,skip_bp_head=args.skip_bp_head)['stages']
    for stage in execution:
        o,k,i=stage['outer'],stage['kind'],stage['indices']
        started=time.monotonic()
        print(json.dumps(dict(event='STAGE_BEGIN',outer=o,stage=k,indices=i)),flush=True)
        result=journal.run(stage_id(o,k,i),lambda o=o,k=k,i=i:backend.execute(o,k,i))
        atomic_write_json(args.output/'progress.json',dict(outer=o,stage=k,indices=i,
            status='COMMITTED',elapsed_seconds=time.monotonic()-started,
            optimizer_steps=result.get('optimizer_steps','NOT_AVAILABLE_SEE_OPTIMIZER_HISTORY'),
            model_hashes=result.get('hashes_after'),latest_pointer=str(journal.pointer)))
        print(json.dumps(dict(event='STAGE_COMMITTED',outer=o,stage=k,indices=i)),flush=True)
        if k in ('data','SDF','cross_block','outer_commit'):
            backend.write_histories(journal.results,figures=k=='outer_commit')
    completed=[o for o in range(12,args.end_outer+1) if stage_id(o,'outer_commit',{}) in journal.completed]
    atomic_write_json(args.output/'final_summary.json',dict(completed_outers=completed,
        TRAINING_COMPLETED=completed==list(range(12,args.end_outer+1)),
        ECONOMIC_EQUILIBRIUM_CONVERGED='NOT_ASSERTED',source_commit=commit,
        BP_HEAD_TRAINING_SKIPPED_BY_USER=args.skip_bp_head,
        SDF_skipped_stages=[k for k,v in journal.results.items() if v.get('status')=='SDF_SKIPPED_NO_VALID_PAIRS'],
        checkpoints=[str(args.output/f'outer_{o:03d}'/'outer_complete.pt') for o in completed]))
    return 0


if __name__=='__main__':raise SystemExit(main())
