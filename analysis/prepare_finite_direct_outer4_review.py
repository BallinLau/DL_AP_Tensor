"""Review-only preparation. Deliberately contains no training execution path.

The user explicitly requires human review before any Slurm/optimizer run.
This command can only construct CPU initial banks and combine saved weights.
It does not pretend the full nested runner has been integrated.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import torch

from analysis.nested_equilibrium_utils import atomic_torch_save,atomic_write_json,tensor_hash,stable_json_hash
from data.finite_direct_sampling import SamplingConfig,build_template_bank,build_or_load_sampling,file_hash

SOURCE_COMMIT='3575abd44dd2b7e197662f2b053db036ff18f76e'
FIRM_HASH='fce66108a07152375c3e4a84950b001d3555c10e2a2c9c7c3540afefa5a93820'
FIRM_FILE_HASH='25183836fc856915448444e32bca997f2ec0e5c894e48f730180efe3a88c2110'


def hash_state(state,prefix=None):
    if prefix is not None:
        state={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        if not state: raise ValueError(f'missing checkpoint component {prefix}')
    return tensor_hash(sorted(state.items()))


def combined_checkpoint(firm_path,fc1_path,source_run):
    source_run=Path(source_run)
    source_meta=json.loads((source_run/'metadata.json').read_text())
    source_summary=json.loads((source_run/'summary.json').read_text())
    if source_meta['source_commit']!=SOURCE_COMMIT: raise RuntimeError('wrong FC1 experiment version')
    if source_summary['MACRO_ITERATIONS_COMPLETED']!=8 or source_summary['status']!='COMPLETED':
        raise RuntimeError('FC1 phi8 experiment not complete')
    if file_hash(firm_path)!=FIRM_FILE_HASH: raise RuntimeError('Firm checkpoint file SHA-256 mismatch')
    original=torch.load(firm_path,map_location='cpu',weights_only=False)
    phi8=torch.load(fc1_path,map_location='cpu',weights_only=False)
    if phi8['iteration']!=8: raise RuntimeError('FC1 checkpoint is not phi8')
    if hash_state(original['models']['policy_value'])!=FIRM_HASH: raise RuntimeError('Firm weight mismatch')
    if hash_state(phi8['models']['policy_value'])!=FIRM_HASH: raise RuntimeError('phi8 frozen Firm mismatch')
    models=deepcopy(original['models']); old_macro=models['sdf_fc1']
    fc1_keys=[k for k in old_macro if k.startswith('fc1_model.')]
    if set(fc1_keys)!={k for k in phi8['models']['sdf_fc1'] if k.startswith('fc1_model.')}:
        raise RuntimeError('FC1 checkpoint parameter-name mismatch')
    for k in fc1_keys:
        v=phi8['models']['sdf_fc1'][k]
        if v.shape!=old_macro[k].shape or v.dtype!=old_macro[k].dtype:
            raise RuntimeError(f'FC1 architecture mismatch: {k}')
        old_macro[k]=v.detach().clone()
    expected=source_summary['hashes_after']
    for name,prefix in [('sdf','sdf_model.'),('wealth','value_model.'),
                        ('fc1_c','fc1_model.FC1_C.'),('fc1_k','fc1_model.FC1_K.')]:
        if hash_state(old_macro,prefix)!=expected[name]: raise RuntimeError(f'combined {name} hash mismatch')
    untouched=[k for k in old_macro if not k.startswith('fc1_model.')]
    if any(not torch.equal(old_macro[k],original['models']['sdf_fc1'][k]) for k in untouched):
        raise RuntimeError('combining FC1 modified SDF/Wealth')
    manifest=dict(firm_source=str(firm_path),firm_source_sha256=file_hash(firm_path),
        firm_source_commit=original.get('git_head'),fc1_source_commit=source_meta['source_commit'],
        fc1_source=str(fc1_path),fc1_source_sha256=file_hash(fc1_path),
        hashes=expected,firm_reused_outer_0=True,
        initialization='NEW_RUN_WEIGHT_WARM_START_NOT_OLD_TRANSACTION_RESUME',
        optimizer_state_policy='DO_NOT_IMPORT_OLD_OPERATOR_OPTIMIZERS; fresh per-stage optimizer on first entry; exact restore required within new stages',
        old_training_banks_imported=False,old_simulation_datasets_imported=False,
        outer_0_operator_transition=True,fc2_must_not_execute=True)
    return dict(format='finite_direct_outer4_initial_weights_v1',models=models,
                source_manifest=manifest,training_performed=False),manifest


def stage_plan():
    rows=[]
    for outer in range(4):
        stages=['reuse_verified_firm'] if outer==0 else ['finite_firm_bank','P0_PI','Q','BP_distillation']
        stages+=['macro_D0','FC1_0','macro_D1_refresh','FC1_1','macro_D2_refresh',
                 'SDF_TRUE_ONLY_chunk0','SDF_TRUE_ONLY_chunk1','cross_block','outer_checkpoint']
        rows.extend(dict(outer=outer,stage=s,completed=False) for s in stages)
    return rows


def prepare(source_run,firm_checkpoint,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    bank,manifest=build_template_bank(source_run,expected_source_commit=SOURCE_COMMIT,expected_firm_hash=FIRM_HASH)
    _,coverage=build_or_load_sampling(output/'sampling',SamplingConfig(),bank,manifest)
    initial,checkpoint_manifest=combined_checkpoint(firm_checkpoint,Path(source_run)/'fc1_iteration_008_complete.pt',source_run)
    path=output/'initial_combined_checkpoint.pt'
    if path.exists():
        previous=torch.load(path,map_location='cpu',weights_only=False)
        if previous['source_manifest']!=initial['source_manifest']:
            raise RuntimeError('existing combined checkpoint provenance mismatch')
        for name in initial['models']:
            if hash_state(previous['models'][name])!=hash_state(initial['models'][name]):
                raise RuntimeError('existing combined checkpoint weights mismatch')
    else: atomic_torch_save(path,initial)
    checkpoint_manifest['combined_sha256']=file_hash(path)
    atomic_write_json(output/'checkpoint_manifest.json',checkpoint_manifest)
    plan=dict(representation='FINITE_FIRM_DIRECT_AGGREGATION',root_search_allowed=False,
        equilibrium_outer_max=4,firm_inner_max=3,p_policy_outer_max=3,p_adam_steps=200,
        p_lbfgs_max_iter=100,q_picard_outer_max=3,q_fit_steps=300,policy_distill_max_steps=1000,
        macro_inner_max=2,fc1_min_epochs=25,fc1_max_epochs=40,fc1_patience=10,
        fc1_C_weight=1.,fc1_K_weight=.25,sdf_chunks=2,sdf_epochs_per_chunk=50,
        sdf_mode='TRUE_ONLY_SIGNED_AIO_AL',stage_plan=stage_plan(),
        sampling_signature=json.loads((output/'sampling/sampling_signature.json').read_text()),
        fc1_selection='fixed training-development subset, not validation',
        validation_reused_across_outers=True,failed_children_retained_in_audit=True,
        fc1_allows_single_valid_child=True,sdf_requires_both_valid_children=True,
        convergence_gates='REPORT_ONLY; do not stop before Outer3 on nonconvergence',
        resource_request=dict(partition='a01',gpus='gpu:a800:1',cpus=8,memory_GB=128,walltime_days=7),
        hard_stops=['nonfinite','corrupt_checkpoint','no_legal_required_training_data','lineage_or_pairing_error'])
    plan['fingerprint']=stable_json_hash(plan)
    atomic_write_json(output/'training_plan.json',plan)
    source=dict(branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty_status=subprocess.check_output(['git','status','--short'],text=True),
        note='Preparation provenance only. No training-data-generating commit exists for a new run.')
    atomic_write_json(output/'preparation_source_manifest.json',source)
    status=dict(IMPLEMENTATION_COMPLETED=True,BETA_EMPIRICAL_SAMPLING_READY=True,
        FOUR_OUTER_RUNNER_READY_FOR_REVIEW=True,FORMAL_TRAINING_STARTED=False,
        SLURM_JOB_SUBMITTED=False,CURRENT_PHASE='WAITING_FOR_USER_REVIEW',
        OPTIMIZER_STEPS_EXECUTED=0,PREPARATION_ONLY=True,
        remaining=['Explicit human review and approval; no formal execution is authorized by preparation.',
            'A800 numerical/performance validation and actual four-Outer optimizer results are NOT tested in this review task.',
            'Firm P/Q/BP restart uncommitted blocks; SDF restarts uncommitted chunks, not exact step continuation.'],
        coverage=coverage)
    atomic_write_json(output/'review_status.json',status)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--firm-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(prepare(args.source_run,args.firm_checkpoint,args.output),indent=2))


if __name__=='__main__': main()
