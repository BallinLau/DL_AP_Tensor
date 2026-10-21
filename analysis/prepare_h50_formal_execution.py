"""Explicit user-authorized CPU certification and one-shot Slurm submission.

No optimizer is stepped here. Approval bytes live outside the clean checkout.
Never retries an uncertain sbatch result: inspect submission_intent first.
"""
import argparse
from dataclasses import asdict
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.run_initial_hstar_h50_broad_macro import (
    H50Backend,approval_binding,require_approval,plan,coverage_config,firm_coverage_config)
from analysis.run_full_nested_finite_direct_outer4 import ProductionBackend
from analysis.finite_direct_continuation_state import verify_optimizer_restored,state_digest
from analysis.h50_source_certification import certify,verify_fixed_validation
from analysis.nested_equilibrium_utils import atomic_write_json,restore_training_transaction
from analysis.run_fc1_self_consistent_macro import hashes
from data.finite_direct_runtime import load_locked_sampling
from data.finite_direct_joint_coverage import JointCoverageConfig
from data.finite_direct_sampling import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment-root',required=True,type=Path)
    p.add_argument('--run-root',required=True,type=Path)
    p.add_argument('--submit',action='store_true')
    cli=p.parse_args()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    deploy=cli.deployment_root;deploy.mkdir(parents=True,exist_ok=True)
    lock=(deploy/'submission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (deploy/'submission_intent.json').exists():raise RuntimeError('Submission already attempted; inspect existing record, never duplicate')
    old=Path('/home/fit/zhuyingz/WORK/LiuHao/cachedir/finite_direct_dynamic_outer4_11_v2_97a0181_r1')
    prep=Path('/home/fit/zhuyingz/WORK/LiuHao/finite_direct_dynamic_final_20261021/prepared_v4_97a018118e917fc5bc5d6b7ed3dd662890560595')
    assert not cli.run_root.exists(),'New run only; use documented original approval for resume'
    pointer=old/'transactions/latest.json'
    journal=old/'transactions'/json.loads(pointer.read_text())['checkpoint']
    cfg=deploy/'coverage_config.json';firm=deploy/'firm_macro_config.json'
    atomic_write_json(cfg,asdict(JointCoverageConfig(candidate_batch_size=16,max_candidates=32768)))
    shutil.copyfile(ROOT/'review/h50_formal_execution/firm_macro_config.json',firm)
    a=SimpleNamespace(sampling_dir=prep/'continuation_sampling',
        initial_checkpoint=old/'outer_011/outer_complete.pt',source_journal_checkpoint=journal,
        source_latest_pointer=pointer,source_artifact_dir=old/'dynamic/outer_011',
        coverage_config=cfg,firm_macro_config=firm,output=cli.run_root,
        end_outer=20,skip_bp_head=True,parent_batch_size=32,fit_parent_batch_size=128,
        firm_batch_size=2048,execute_approved=True,approval=deploy/'approval_v5.json',device='cpu')
    learning,_,_,_,audits=certify(a.initial_checkpoint,journal,pointer,source_job=702636,artifact_dir=a.source_artifact_dir)
    verify_fixed_validation(a.sampling_dir)
    # Actual production restore, no simulation or optimization. This verifies
    # persistent optimizer/scheduler/AL and Fresh Pair restoration, not weights only.
    # CUDA Philox generator bytes are not a CPU MT19937 generator state.
    # Do not call restore_fresh on CPU, convert bytes, or seed a replacement.
    # The unchanged H50Backend restores the exact saved bytes on CUDA at launch.
    a.initial_hstar_h50_adapter=True
    backend=ProductionBackend(a,load_locked_sampling(a.sampling_dir),learning_snapshot=learning)
    optimizers={**backend.episode.optimizers,'fc1_conditional':backend.fc1}
    restore_training_transaction(learning['transaction'],models=backend.models,
        optimizers=optimizers,episode=backend.episode)
    optimizer_audit={name:verify_optimizer_restored(learning['transaction']['optimizers'][name],opt)
        for name,opt in optimizers.items()}
    assert all(x['verified'] for x in audits.values())
    before=hashes(backend.models)
    audit=dict(source_commit=commit,certification=audits,
        restored_optimizer_audit=optimizer_audit,
        fresh_pair_state_sha256=state_digest(learning['fresh']['pair_rng']),
        fresh_pair_restore='ORIGINAL_CUDA_BYTES_PRESERVED; ACTUAL_RESTORE_AT_CUDA_START',
        restored_model_hashes=before,training_performed=False,optimizer_steps=0,
        fixed_validation_verified=True,phase='CPU_PREPARATION_PASS')
    atomic_write_json(deploy/'preflight.json',audit)
    expected=approval_binding(a,commit)
    approval=dict(expected,approved=True,approved_by='user',
        approved_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorization='User attached explicit TRAINING EXECUTION REQUEST 5d3d25b8: H50 Outer12–20, Hybrid 0.5, skip BP Head, unchanged economic rules.')
    atomic_write_json(a.approval,approval)
    assert require_approval(a,commit)
    atomic_write_json(deploy/'training_plan.json',plan(coverage_config(a),firm_coverage_config(a),end_outer=20,skip_bp_head=True))
    env=dict(os.environ)
    env.update(REPO_DIR=str(ROOT),REQUIRED_COMMIT=commit,RUN_ROOT=str(cli.run_root),
        SAMPLING_DIR=str(a.sampling_dir),INITIAL_CHECKPOINT=str(a.initial_checkpoint),
        SOURCE_JOURNAL_CHECKPOINT=str(journal),SOURCE_LATEST_POINTER=str(pointer),
        SOURCE_ARTIFACT_DIR=str(a.source_artifact_dir),USER_APPROVAL_JSON=str(a.approval),
        COVERAGE_CONFIG_JSON=str(cfg),FIRM_MACRO_CONFIG_JSON=str(firm),RESUME='0')
    env.pop('CUDA_VISIBLE_DEVICES',None)
    atomic_write_json(deploy/'slurm_environment.json',{k:env[k] for k in (
        'REPO_DIR','REQUIRED_COMMIT','RUN_ROOT','SAMPLING_DIR','INITIAL_CHECKPOINT',
        'SOURCE_JOURNAL_CHECKPOINT','SOURCE_LATEST_POINTER','SOURCE_ARTIFACT_DIR',
        'USER_APPROVAL_JSON','COVERAGE_CONFIG_JSON','FIRM_MACRO_CONFIG_JSON','RESUME')})
    script=ROOT/'slurm/run_initial_hstar_h50_broad_macro_a800.slurm'
    subprocess.run(['bash','-n',str(script)],check=True)
    command=['/rmprog/slurm/v24.05.1/bin/sbatch','--parsable',
        '--output='+str(cli.run_root/'logs/slurm.%j.out'),
        '--error='+str(cli.run_root/'logs/slurm.%j.err'),str(script)]
    atomic_write_json(deploy/'planned_submission.json',dict(command=command,source_commit=commit,
        approval_sha256=file_hash(a.approval),approval_binding_verified=True))
    if not cli.submit:
        print('CPU_PREPARATION_PASS; NOT_SUBMITTED',flush=True);return
    (cli.run_root/'logs').mkdir(parents=True,exist_ok=False)
    atomic_write_json(deploy/'submission_intent.json',dict(command=command,source_commit=commit,run_root=str(cli.run_root)))
    result=subprocess.run(command,env=env,text=True,capture_output=True)
    atomic_write_json(deploy/'submission_response.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    if result.returncode:raise RuntimeError(result.stderr)
    job=result.stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Uncertain submission outcome; inspect Slurm, do not retry')
    atomic_write_json(deploy/'slurm_job.json',dict(job_id=job,run_root=str(cli.run_root),source_commit=commit))
    print(json.dumps(dict(SLURM_JOB_ID=job,CPU_PREPARATION='PASS',TRAINING_COMPLETED=False)),flush=True)


if __name__=='__main__':main()
