#!/usr/bin/env python3
"""Four-Outer finite-direct runner. Defaults to a NON-EXECUTING plan.

Production execution requires BOTH --execute-approved and a separately supplied
approval JSON bound to this exact plan and clean source commit. This task does
not create approval or run this mode. Firm blocks and SDF chunks restart from
their committed beginning after interruption; FC1 commits every full epoch.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass,asdict
import json
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.nested_equilibrium_utils import (atomic_torch_save,atomic_write_json,module_hash,
    stable_json_hash,capture_training_transaction,restore_training_transaction,seed_all,require_finite_module,
    capture_rng_state,restore_rng_state)
from analysis.finite_direct_outer4_state import StageJournal
from data.finite_direct_runtime import (OPERATOR,load_locked_sampling,generate_dataset,
    firm_bank_from_captures,conditional_child_bank,check_firm_lineage,verify_cache,firm_bp_parent_rows)
from data.finite_direct_macro_contract import select_training_rows,production_sdf_tensors
from data.finite_direct_sampling import file_hash
from analysis.economic_config import AnalysisEconomicConfig
from analysis.run_fc1_self_consistent_macro import (setup,hashes,xy,production_loss,evaluate,coverage,install_guards)
from analysis.run_full_nested_equilibrium import (_p_policy_outer,_q_picard_outer,_distill_policy_heads,
    build_problem,evaluate_problem,build_q_survival_problem,evaluate_q_problem)
from analysis.test_frozen_policy_root_solve import P_PREFIXES,Q_PREFIXES
from training.episode import SDFTrainingPhase
from training.sdf_shock_bank import SDFShockBank,_make_generator
from training.inner_fixed_point import InnerFPCollocationBank,InnerFPChildBank
from analysis.finite_direct_outer_metrics import economic_metrics,write_histories


@dataclass(frozen=True)
class Budget:
    equilibrium_outer_max:int=4
    firm_inner_max:int=3
    p_policy_outer_max:int=3
    p_adam_steps:int=200
    p_lbfgs_max_iter:int=100
    q_picard_outer_max:int=3
    q_fit_steps:int=300
    policy_distill_max_steps:int=1000
    macro_inner_max:int=2
    fc1_min_epochs:int=25
    fc1_max_epochs:int=40
    fc1_patience:int=10
    sdf_chunks:int=2
    sdf_epochs_per_chunk:int=50


def stages(budget=Budget()):
    for outer in range(budget.equilibrium_outer_max):
        yield outer,'outer_start',{}
        if outer==0: yield outer,'reuse_firm',{}
        else:
            yield outer,'firm_bank',{}
            for inner in range(budget.firm_inner_max):
                for p in range(budget.p_policy_outer_max): yield outer,'P',dict(inner=inner,policy=p)
                for q in range(budget.q_picard_outer_max): yield outer,'Q',dict(inner=inner,picard=q)
                yield outer,'BP',dict(inner=inner)
        yield outer,'data',dict(version=0)
        for macro in range(budget.macro_inner_max):
            for epoch in range(1,budget.fc1_max_epochs+1): yield outer,'FC1',dict(macro=macro,epoch=epoch)
            yield outer,'data',dict(version=macro+1)
        for chunk in range(budget.sdf_chunks): yield outer,'SDF',dict(chunk=chunk)
        yield outer,'post_sdf_validation',{}
        yield outer,'cross_block',{}
        yield outer,'outer_commit',{}


def stage_id(outer,kind,indices):
    suffix='_'.join(f'{k}{v}' for k,v in indices.items())
    return f'outer_{outer:03d}/{kind}'+('/'+suffix if suffix else '')


def training_plan():
    return dict(operator=OPERATOR,budget=asdict(Budget()),train_parents=4096,validation_parents=1024,
        firms=200,children=2,empirical_semantics='RECORDED_INITIAL_PARENT_DISTRIBUTION',
        execution=[dict(id=stage_id(o,k,i),outer=o,kind=k,indices=i) for o,k,i in stages()],
        root_search=False,fc2=False,firm_resume='RESTART_UNCOMMITTED_BLOCK',
        fc1_resume='COMMITTED_EPOCH_MATCHED_MODEL_OPTIMIZER',sdf_resume='RESTART_UNCOMMITTED_CHUNK',
        convergence_is_diagnostic_only=True)


def require_approval(args,plan,commit):
    if not args.execute_approved: return False
    if args.approval is None: raise RuntimeError('explicit user approval file required')
    approval=json.loads(args.approval.read_text())
    head=dict(schema_version=2,source_commit=commit,plan_fingerprint=stable_json_hash(plan))
    if approval.get('approved') is not True or any(approval.get(k)!=v for k,v in head.items()):
        raise RuntimeError('approval does not match exact source/plan/input artifacts')
    signature_path=args.sampling_dir/'sampling_signature.json'
    signature=json.loads(signature_path.read_text())
    expected=dict(**head,
        sampling_signature_fingerprint=signature['fingerprint'],
        sampling_signature_sha256=file_hash(signature_path),
        initial_checkpoint_sha256=file_hash(args.initial_checkpoint))
    # The signature file SHA also binds its per-artifact hash manifest: changing
    # the bank hashes while retaining a configuration fingerprint cannot pass.
    if approval.get('approved') is not True or any(approval.get(k)!=v for k,v in expected.items()):
        raise RuntimeError('approval does not match exact source/plan/input artifacts')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('formal execution requires clean source checkout')
    return True


def sdf_batches(frame,batch_size,device,offset=0):
    tensors=production_sdf_tensors(frame,device=device)
    batches=[]
    for start in range(0,len(tensors['parents']),batch_size):
        end=start+batch_size
        batches.append(dict(parent=tensors['parents'][start:end],
            children=[c[start:end] for c in tensors['children']],
            parent_index=tensors['parent_index'][start:end]+offset))
    return batches,tensors['lineage']


def fresh_state(episode):
    names=['_sdf_shock_bank_n_parents','_sdf_shock_bank_epoch','_sdf_shock_bank_episode_id',
           '_sdf_shock_bank_key','_current_epoch_idx','_run_batches_epoch_offset','step_count',
           'sdf_fc1_step_count','bp_distill_step_count','_nonfinite_grad_streak','_nonfinite_grad_total']
    state={name:deepcopy(getattr(episode,name,None)) for name in names}
    bank=getattr(episode,'_sdf_shock_bank',None)
    state['bank']=None if bank is None else dict(eps=bank.eps.detach().cpu().clone(),bank_size=bank.bank_size,
                                                refresh_id=bank.refresh_id,base_seed=bank.base_seed)
    gen=getattr(episode,'_sdf_pair_generator',None)
    state['pair_rng']=None if gen is None else gen.get_state().cpu()
    return state


def restore_fresh(episode,state):
    for name,value in state.items():
        if name not in ('bank','pair_rng') and value is not None: setattr(episode,name,deepcopy(value))
    b=state['bank']; episode._sdf_shock_bank=None
    if b is not None: episode._sdf_shock_bank=SDFShockBank(**{**b,'eps':b['eps'].to(episode.device)})
    episode._sdf_pair_generator=None
    if state['pair_rng'] is not None:
        episode._sdf_pair_generator=_make_generator(episode.device)
        episode._sdf_pair_generator.set_state(state['pair_rng'].cpu())


def set_scope(models,scope):
    for model in models.values(): model.requires_grad_(False); model.eval()
    if scope=='firm': models['policy_value'].requires_grad_(True)
    if scope=='fc1': models['sdf_fc1'].fc1_model.requires_grad_(True)
    if scope=='sdf':
        models['sdf_fc1'].sdf_model.requires_grad_(True)
        models['sdf_fc1'].value_model.requires_grad_(True)


@contextmanager
def finite_gradient_guard(model):
    def check(g):
        if not torch.isfinite(g).all(): raise FloatingPointError('nonfinite production gradient')
        return g
    handles=[p.register_hook(check) for p in model.parameters() if p.requires_grad]
    try: yield
    finally:
        for h in handles: h.remove()


class ProductionBackend:
    def __init__(self,args,bank_data,*,learning_snapshot=None):
        self.args=args; self.root=args.output; self.banks,self.schedules,self.signature=bank_data
        models,episode,fc1,guard=setup(argparse.Namespace(device=args.device,checkpoint=args.initial_checkpoint),
            **({} if learning_snapshot is None else {'initial_models':learning_snapshot['transaction']['models']}))
        self.models=models; self.episode=episode; self.fc1=fc1; self.fc2_guard=guard
        if learning_snapshot is None:
            cp=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=False)
            if cp.get('format')!='finite_direct_outer4_initial_weights_v1' or cp.get('training_performed') is not False:
                raise RuntimeError('initial checkpoint is not the reviewed combined weight warm start')
            if hashes(models)!=cp['source_manifest']['hashes']:
                raise RuntimeError('initial combined component hash mismatch')
        episode._fc1_firms=200; self.budget=Budget()
        hp=episode.hyperparams
        hp.bp_distill_max_optimizer_steps=1000; hp.bp_distill_epochs=1000
        hp.bp_distill_grad_clip_norm=10.; hp.pv_bp_head_training_enabled=True
        # Preserve production Signed AiO, normalized residual, AL and all clamps.
        if hp.sdf_wealth_loss_mode!='signed_aio' or hp.sdf_wealth_residual_mode!='normalized_ratio':
            raise RuntimeError('expected production normalized Signed AiO')
        hp.sdf_al_reset_on_true_start=False
        self.state=dict(data={},firm_banks={},fc1_progress={},outer_metrics={},sdf_epoch=0)
        self.solver_args=argparse.Namespace(p_adam_steps=200,p_adam_lr=1e-4,p_eval_every=10,
            p_lbfgs_max_iter=100,p_lbfgs_chunk_iterations=20,q_fit_steps=300,q_lr=1e-4,
            policy_distill_max_epochs=1000,policy_distill_max_steps=1000)
        self.economic=AnalysisEconomicConfig.from_current_config()
        self.counter={'ROOT_SEARCH_CALLS':0,'FAST_CLEARING_CALLS':0}
        if not getattr(args,'initial_hstar_h50_adapter',False):
            install_guards(self.counter)

    def snapshot(self):
        return dict(transaction=capture_training_transaction(models=self.models,
            optimizers={**self.episode.optimizers,'fc1_conditional':self.fc1},episode=self.episode),
            fresh=fresh_state(self.episode),state=deepcopy(self.state))

    def restore(self,snapshot):
        restore_training_transaction(snapshot['transaction'],models=self.models,
            optimizers={**self.episode.optimizers,'fc1_conditional':self.fc1},episode=self.episode)
        restore_fresh(self.episode,snapshot['fresh']); self.state=deepcopy(snapshot['state'])
        # Each committed Firm block uses the CURRENT online policy as the
        # simulation Grid target. Reconstruct that deterministic hard snapshot.
        self.episode.refresh_firm_target()

    def _data(self,outer,version,split):
        key=f'{outer}/D{version}/{split}'; record=self.state['data'][key]
        path=Path(record['path']); meta=path/'manifest.json'
        if file_hash(meta)!=record['manifest_sha256']: raise RuntimeError('dataset manifest changed after commit')
        manifest=json.loads(meta.read_text())
        if file_hash(path/'observations.csv')!=manifest['observations_sha256']: raise RuntimeError('dataset content corruption')
        return pd.read_csv(path/'observations.csv')

    def _generate(self,outer,stage,*,capture=True):
        target=self.episode._target_policy_value()
        if module_hash(target)!=module_hash(self.models['policy_value']):
            raise RuntimeError('simulation Grid target is stale relative to committed Firm')
        result={}
        for split in ('train','validation'):
            path=self.root/f'outer_{outer:03d}'/stage/split
            frame,snaps,manifest=generate_dataset(self.models,self.episode,self.banks[split],self.schedules[split],
                self.signature,path,outer=outer,stage=stage,parent_batch_size=self.args.parent_batch_size,capture=capture)
            result[split]=(frame,snaps,manifest,path)
        return result

    def _firm(self,outer):
        record=self.state['firm_banks'][str(outer)]; path=Path(record['path'])
        if file_hash(path)!=record['sha256']: raise RuntimeError('conditional Firm bank corruption')
        return torch.load(path,map_location='cpu',weights_only=False)

    def _conditional(self,bank,seed):
        return conditional_child_bank(self.models['sdf_fc1'],bank,seed=seed,n_child_shocks=2,
            economic=self.economic,device=self.episode.device,m_min=self.episode.hyperparams.pv_m_clamp_min,
            m_max=self.episode.hyperparams.pv_m_clamp_max)

    def execute(self,outer,kind,indices):
        before=hashes(self.models); result=self._execute(outer,kind,indices)
        if kind in ('P','Q','BP'):
            # Existing solvers optimize explicit online models outside
            # Episode.train_step, so its usual epoch-hard refresh is not called.
            self.episode.refresh_firm_target()
        after=hashes(self.models)
        allowed={'P':{'firm'},'Q':{'firm'},'BP':{'firm'},'FC1':{'fc1_c','fc1_k'},'SDF':{'sdf','wealth'}}.get(kind,set())
        for name in before:
            if name not in allowed and before[name]!=after[name]: raise RuntimeError(f'{kind} violated frozen {name}')
        for name,model in self.models.items(): require_finite_module(name,model)
        if self.fc2_guard['calls'] or any(self.counter.values()): raise RuntimeError('root/FC2 guard violated')
        result.update(hashes_before=before,hashes_after=after,stage=kind,outer=outer,indices=indices)
        return result

    def _execute(self,outer,kind,i):
        directory=self.root/f'outer_{outer:03d}'; directory.mkdir(parents=True,exist_ok=True)
        if kind=='outer_start':
            self.state['outer_metrics'][str(outer)]={'initial_hashes':hashes(self.models)}
            if outer:
                prev=self.state['outer_metrics'][str(outer-1)]['committed_hashes']
                if hashes(self.models)!=prev: raise RuntimeError('next Outer did not load previous committed Macro')
            return dict(previous_macro_verified=True,operator_transition=(outer==0))
        if kind=='reuse_firm':
            return dict(OUTER_0_FIRM_REUSED=True,OUTER_0_FIRM_SOURCE_OPERATOR='ROOT_CLEARED',
                OUTER_0_MACRO_OPERATOR='FINITE_FIRM_DIRECT_AGGREGATION',OUTER_0_OPERATOR_TRANSITION=True,
                optimizer_steps=0)
        if kind=='firm_bank':
            data=self._generate(outer,'firm_source'); banks={}; children={}
            for split,(_,snaps,_,_) in data.items():
                banks[split]=firm_bank_from_captures(snaps,split=split,max_rows=8192 if split=='train' else 2048,
                    seed=20261020+outer*1000+(0 if split=='train' else 100000))
                children[split]=self._conditional(banks[split],banks[split].seed+1)
            check_firm_lineage(banks['train'],banks['validation'])
            path=directory/'firm_conditional_bank.pt'
            atomic_torch_save(path,dict(banks={s:b.payload() for s,b in banks.items()},
                children={s:b.payload() for s,b in children.items()},macro_hash=module_hash(self.models['sdf_fc1'])))
            self.state['firm_banks'][str(outer)]=dict(path=str(path),sha256=file_hash(path))
            # Fixed independent evaluation rows from Outer1 remain fixed for all
            # later cross-block probes. Outer0 builds its own transition probe.
            if 'probe_bank' not in self.state: self.state['probe_bank']=self.state['firm_banks'][str(outer)]
            return dict(train_rows=len(banks['train'].parent_states),validation_rows=len(banks['validation'].parent_states))
        if kind in ('P','Q','BP'):
            payload=self._firm(outer); macro=module_hash(self.models['sdf_fc1'])
            if payload['macro_hash']!=macro: raise RuntimeError('stale current-M conditional Firm bank')
            banks={s:InnerFPCollocationBank(**{k:v for k,v in p.items() if k!='fingerprint'}) for s,p in payload['banks'].items()}
            children={s:p['children'].to(self.episode.device) for s,p in payload['children'].items()}
            parents={s:b.parent_states.to(self.episode.device) for s,b in banks.items()}
            set_scope(self.models,'firm'); model=self.models['policy_value']
            kwargs=dict(model=model,episode=self.episode,train_parents=parents['train'],train_children=children['train'],
                validation_parents=parents['validation'],validation_children=children['validation'],
                batch_size=self.args.firm_batch_size,args=self.solver_args)
            if kind=='P':
                solved,row,history=_p_policy_outer(**kwargs,policy_outer=i['policy'])
                model.load_state_dict(solved.state_dict()); del solved
            elif kind=='Q': row,history=_q_picard_outer(**kwargs,picard_outer=i['picard'])
            else:
                # Parent[7] would force production BP M=1. Attach observed true
                # context through metadata-aware batching below, not fake rows.
                row=self._bp(banks,payload['children'],directory/f'firm_{i["inner"]:02d}_BP')
                history=row.get('history',[])
            history=pd.DataFrame(history)
            if kind in ('P','Q'):
                row['optimizer_history_rows']=len(history)
                row['optimizer_executed']=len(history)>0
                if not len(history): raise RuntimeError(f'{kind} executed no optimizer updates')
            history.to_csv(directory/(stage_id(outer,kind,i).replace('/','_')+'.csv'),index=False)
            return row
        if kind=='data':
            version=i['version']; generated=self._generate(outer,f'D{version}')
            result={}
            for split,(frame,snaps,_,path) in generated.items():
                rows=select_training_rows(frame)
                self.state['data'][f'{outer}/D{version}/{split}']=dict(path=str(path),manifest_sha256=file_hash(path/'manifest.json'))
                result[split]=coverage(frame,len(self.banks[split]['state']['x']))
                result[split].update(evaluate(self.models['sdf_fc1'],frame,self.episode.device))
                result.setdefault('economic_metrics',[]).extend(economic_metrics(
                    frame,self.schedules[split],outer=outer,stage=f'D{version}'))
                if not len(rows.fc1): raise RuntimeError('NO_LEGAL_FC1_LABELS')
            # Build fixed evaluation bank from initial models/CRN even in Outer0.
            if 'probe_bank' not in self.state:
                banks={s:firm_bank_from_captures(x[1],split=s,max_rows=512,seed=7321+(s=='validation')) for s,x in generated.items()}
                check_firm_lineage(banks['train'],banks['validation'])
                children={s:self._conditional(b,b.seed+1) for s,b in banks.items()}
                path=self.root/'fixed_probe_bank.pt'
                atomic_torch_save(path,dict(banks={s:b.payload() for s,b in banks.items()},
                    children={s:c.payload() for s,c in children.items()}))
                self.state['probe_bank']=dict(path=str(path),sha256=file_hash(path))
            if version==0:
                result['pre_macro_firm_residual']=self._cross(outer,label='pre_macro')
            return result
        if kind=='FC1': return self._fc1_epoch(outer,i)
        if kind=='SDF': return self._sdf_chunk(outer,i['chunk'])
        if kind=='post_sdf_validation': return self._post_sdf_validation(outer)
        if kind=='cross_block': return self._cross(outer)
        if kind=='outer_commit':
            self.state['outer_metrics'][str(outer)]['committed_hashes']=hashes(self.models)
            path=directory/'outer_complete.pt'; atomic_torch_save(path,self.snapshot())
            return dict(checkpoint=str(path),sha256=file_hash(path),completed=True,
                convergence='NOT_ASSERTED_FROM_TRAINING_LOSS')
        raise ValueError(kind)

    @torch.no_grad()
    def _post_sdf_validation(self,outer):
        stage='POST_SDF_VALIDATION_RESIMULATION'; split='validation'
        # Never read self._data(outer,2,...): D2 predates the SDF optimizer.
        # Dedicated stage/path plus all current network hashes prevent reuse.
        path=self.root/f'outer_{outer:03d}'/stage/split
        before=hashes(self.models)
        modes={m:m.training for model in self.models.values() for m in model.modules()}
        rng=capture_rng_state()
        try:
            for model in self.models.values(): model.eval()
            frame,_,manifest=generate_dataset(self.models,self.episode,self.banks[split],self.schedules[split],
                self.signature,path,outer=outer,stage=stage,parent_batch_size=self.args.parent_batch_size,capture=False)
        finally:
            for module,mode in modes.items(): module.training=mode
            restore_rng_state(rng)
        if hashes(self.models)!=before: raise RuntimeError('post-SDF evaluation mutated model')
        return dict(evaluation=stage,split=split,path=str(path),manifest_sha256=file_hash(path/'manifest.json'),
            crn_hash=manifest['provenance']['crn_hash'],
            economic_metrics=economic_metrics(frame,self.schedules[split],outer=outer,stage=stage))

    def write_histories(self,results,*,figures=False):
        write_histories(self.root,results,figures=figures)

    def _bp(self,banks,child_payloads,output):
        # Use actual production BP cache/distillation; minimal parent adapter
        # preserves 7D network inputs while exposing raw M to Episode.
        def batches(split):
            parent=firm_bp_parent_rows(banks[split])
            child=child_payloads[split]['children']; b=self.args.firm_batch_size
            return [dict(parent=parent[j:j+b].to(self.episode.device),
                children=[x.to(self.episode.device) for x in child[j:j+b].unbind(1)]) for j in range(0,len(parent),b)]
        model=self.models['policy_value']; before=model.state_dict()
        invariant={k:v.detach().clone() for k,v in before.items() if any(k.startswith(p+'.') for p in P_PREFIXES+Q_PREFIXES)}
        teacher=deepcopy(model).eval().requires_grad_(False)
        train=self.episode._build_bp_target_cache(batches('train'),teacher)
        val=self.episode._build_bp_target_cache(batches('validation'),teacher)
        result=self.episode._run_bp_distillation_stage(train,val,teacher,n_epochs=1000)
        if result.get('optimizer_steps',0)<=0: raise RuntimeError('NO_LEGAL_ACTIVE_BP_TRAINING_DATA_OR_NO_BP_UPDATE')
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in invariant.items()): raise RuntimeError('BP changed non-BP parameters')
        output.mkdir(parents=True,exist_ok=True); atomic_write_json(output/'summary.json',result)
        return result

    def _fc1_epoch(self,outer,i):
        key=f'{outer}/{i["macro"]}'; prog=self.state['fc1_progress'].setdefault(key,dict(best=float('inf'),bad=0,stopped=False,steps=0))
        if prog['stopped']: return dict(skipped='DECLARED_FC1_PATIENCE_AFTER_MIN_EPOCHS',optimizer_steps=0)
        frame=self._data(outer,i['macro'],'train'); eligible=select_training_rows(frame).fc1
        ids=np.sort(eligible.parent_candidate_id.unique()); dev_ids=set(ids[::10])
        dev=eligible[eligible.parent_candidate_id.isin(dev_ids)]; fit=eligible[~eligible.parent_candidate_id.isin(dev_ids)]
        if not len(fit) or not len(dev): raise RuntimeError('NO_LEGAL_FC1_FIT_OR_DEV')
        set_scope(self.models,'fc1'); self.episode.set_sdf_training_phase(SDFTrainingPhase.FC1_ONLY)
        X,Y,f=xy(fit,self.episode.device); D,T,_=xy(dev,self.episode.device)
        parents=torch.tensor(f.parent_candidate_id.to_numpy(),device=self.episode.device)
        unique=torch.unique(parents); order=unique[torch.randperm(len(unique),device=self.episode.device)]
        steps=0; loss_sum=0.; count=0
        with finite_gradient_guard(self.models['sdf_fc1'].fc1_model):
            for start in range(0,len(order),self.args.fit_parent_batch_size):
                chosen=order[start:start+self.args.fit_parent_batch_size]; mask=torch.isin(parents,chosen)
                self.fc1.zero_grad(set_to_none=True); loss=production_loss(self.episode,X[mask],Y[mask])
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite FC1 loss')
                loss.backward(); self.fc1.step(); steps+=1
                count+=int(mask.sum()); loss_sum+=float(loss.detach())*int(mask.sum())
        if count!=len(f): raise RuntimeError('incomplete FC1 epoch')
        with torch.no_grad(): dev_loss=float(production_loss(self.episode,D,T))
        if not np.isfinite(dev_loss): raise FloatingPointError('nonfinite FC1 development loss')
        prog['steps']+=steps
        if dev_loss<prog['best']:
            prog.update(best=dev_loss,bad=0)
            atomic_torch_save(self.root/f'outer_{outer:03d}'/f'FC1_{i["macro"]}_best_diagnostic.pt',
                dict(models=self.models['sdf_fc1'].fc1_model.state_dict(),optimizer=self.fc1.state_dict(),epoch=i['epoch']))
        else: prog['bad']+=1
        prog['stopped']=i['epoch']>=25 and prog['bad']>=10
        return dict(epoch=i['epoch'],optimizer_steps=steps,train_loss=loss_sum/count,dev_loss=dev_loss,
            selection='LAST_EPOCH_MATCHED_OPTIMIZER; BEST_DEV_DIAGNOSTIC_ONLY',
            validation=evaluate(self.models['sdf_fc1'],self._data(outer,i['macro'],'validation'),self.episode.device))

    def _sdf_chunk(self,outer,chunk,*,allow_empty_validation=False):
        set_scope(self.models,'sdf'); e=self.episode; e.add_FC1loss=False
        e.set_sdf_training_phase(SDFTrainingPhase.SDF_TRUE_ONLY)
        train,lineage=sdf_batches(self._data(outer,2,'train'),self.args.fit_parent_batch_size,e.device)
        validation_frame=self._data(outer,2,'validation')
        val=[] if allow_empty_validation and validation_frame.empty else sdf_batches(validation_frame,self.args.fit_parent_batch_size,e.device,offset=len(lineage))[0]
        e._require_sdf_al_outer_loop(SDFTrainingPhase.SDF_TRUE_ONLY)
        e._validate_sdf_al_semantics(SDFTrainingPhase.SDF_TRUE_ONLY)
        if chunk==0: e._configure_sdf_lr_for_phase()
        rows=[]; start=e.sdf_fc1_step_count
        with finite_gradient_guard(self.models['sdf_fc1']):
            for local in range(self.budget.sdf_epochs_per_chunk):
                epoch=self.state['sdf_epoch']; e._run_batches_epoch_offset=epoch
                e._prepare_sdf_shock_bank_for_epoch(train+val,epoch,['sdf_fc1'])
                e._run_batches(train,n_epochs=1,log_interval=100,train_modules=['sdf_fc1'],
                    desc_prefix=f'O{outer} SDF {chunk} ',configure_sdf_lr=False)
                constraints=e._estimate_sdf_al_constraints(train)
                dual=False
                if (epoch+1)%int(e.hyperparams.sdf_al_primal_epochs_per_dual_update)==0:
                    e._update_sdf_al_duals(constraints); dual=True
                validation=(e._evaluate_sdf_fc1_batches(val,prefix=f'outer{outer}_sdf{chunk}_epoch{epoch}',max_batches=None)
                    if val else dict(status='NOT_AVAILABLE_NO_VALID_VALIDATION_PAIRS'))
                self.state['sdf_epoch']+=1
                rows.append(dict(epoch=epoch,constraints=constraints,al=e._get_sdf_al_state(),dual_update=dual,
                                 validation=validation))
        if e.sdf_fc1_step_count<=start: raise RuntimeError('SDF chunk executed no optimizer steps')
        # Production forward-only validation. No strict-convergence early stop.
        metrics=rows[-1]['validation']
        history_path=self.root/f'outer_{outer:03d}'/f'sdf_chunk{chunk}_history.json'
        atomic_write_json(history_path,rows)
        return dict(optimizer_steps=e.sdf_fc1_step_count-start,epochs=self.budget.sdf_epochs_per_chunk,validation=metrics,
                    al_history_path=str(history_path),al_history_sha256=file_hash(history_path),
                    al=e._get_sdf_al_state(),fresh_bank_identity=stable_json_hash(fresh_state(e)))

    def _cross(self,outer,label='post_macro'):
        record=self.state['probe_bank']; path=Path(record['path'])
        if file_hash(path)!=record['sha256']: raise RuntimeError('fixed probe corruption')
        payload=torch.load(path,map_location='cpu',weights_only=False)
        p=payload['banks']['validation']; bank=InnerFPCollocationBank(**{k:v for k,v in p.items() if k!='fingerprint'})
        # SAME parents/context/CRN, refreshed CURRENT Macro M, not stale M.
        seed=payload['children']['validation']['seed']; child=self._conditional(bank,seed)
        parents=bank.parent_states.to(self.episode.device); children=child.children.to(self.episode.device)
        set_scope(self.models,'frozen'); model=self.models['policy_value']
        problem=build_problem(self.episode,model,parents,children,self.args.firm_batch_size,require_legacy_m1=False)
        pmetrics=evaluate_problem(model,problem)
        qproblem,regimes=build_q_survival_problem(episode=self.episode,initial_model=model,frozen_p_model=model,
            parents=parents,children=children,batch_size=self.args.firm_batch_size,m_mode='current_sdf_child',include_all_regimes=True)
        qmetrics=evaluate_q_problem(model,self.episode,qproblem)
        result=dict(P=pmetrics,Q=qmetrics,regimes=regimes,probe_kind='FIXED_7D_PROBE',label=label,fixed_parent_hash=bank.fingerprint,
                    current_macro=module_hash(self.models['sdf_fc1']))
        atomic_write_json(self.root/f'outer_{outer:03d}'/f'cross_block_{label}.json',result)
        return result


def drive(backend,journal,budget=Budget()):
    for outer,kind,indices in stages(budget):
        journal.run(stage_id(outer,kind,indices),lambda o=outer,k=kind,i=indices:backend.execute(o,k,i))
        if hasattr(backend,'write_histories') and kind in ('data','SDF','post_sdf_validation','cross_block','outer_commit'):
            backend.write_histories(journal.results,figures=kind in ('post_sdf_validation','outer_commit'))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sampling-dir',type=Path,required=True)
    p.add_argument('--initial-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--parent-batch-size',type=int,default=128)
    p.add_argument('--fit-parent-batch-size',type=int,default=128)
    p.add_argument('--firm-batch-size',type=int,default=512)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--execute-approved',action='store_true')
    p.add_argument('--approval',type=Path)
    return p.parse_args(argv)


def main(argv=None):
    args=parse_args(argv); plan=training_plan()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if not require_approval(args,plan,commit):
        print(json.dumps(dict(CURRENT_PHASE='WAITING_FOR_USER_REVIEW',FORMAL_TRAINING_STARTED=False,
            plan_fingerprint=stable_json_hash(plan),plan=plan),indent=2)); return
    if min(args.parent_batch_size,args.fit_parent_batch_size,args.firm_batch_size)<=0: raise ValueError('invalid batch size')
    locked=load_locked_sampling(args.sampling_dir)
    if locked[2]['config']['train_parents']!=4096 or locked[2]['config']['validation_parents']!=1024 or locked[2]['config']['firms']!=200:
        raise RuntimeError('formal logical data budget changed')
    seed_all(20261020)
    backend=ProductionBackend(args,locked)
    contract=dict(source_commit=commit,operator=OPERATOR,plan_fingerprint=stable_json_hash(plan),
        economic=asdict(backend.economic),hyperparameters=asdict(backend.episode.hyperparams),
        sampling_signature=locked[2],initial_checkpoint_sha256=file_hash(args.initial_checkpoint),
        runtime=dict(device=args.device,parent_batch=args.parent_batch_size,fit_batch=args.fit_parent_batch_size,firm_batch=args.firm_batch_size))
    journal=StageJournal(args.output/'transactions',contract,backend,resume=args.resume)
    drive(backend,journal)
    atomic_write_json(args.output/'final_summary.json',dict(FOUR_OUTER_TRAINING_COMPLETED=True,
        ECONOMIC_EQUILIBRIUM_CONVERGED='NOT_ASSERTED',source_commit=commit,completed=journal.completed,
        final_hashes=hashes(backend.models)))


if __name__=='__main__': main()
