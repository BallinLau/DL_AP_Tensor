"""CPU-only input certification and pending V6 review materials; never submit."""
import argparse,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.run_h10_h100_longrun_macro import approval_binding,plan,coverage_config,firm_coverage_config
from analysis.h50_source_certification import certify,verify_fixed_validation
from analysis.nested_equilibrium_utils import atomic_write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sampling-dir','initial-checkpoint','source-journal-checkpoint','source-latest-pointer','source-artifact-dir','run-root','review-output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('clean source required')
    certify(a.initial_checkpoint,a.source_journal_checkpoint,a.source_latest_pointer,source_job=702636,artifact_dir=a.source_artifact_dir)
    verify_fixed_validation(a.sampling_dir)
    args=SimpleNamespace(**vars(a),coverage_config=None,firm_macro_config=None,
        end_outer=15,skip_bp_head=True,parent_batch_size=32,fit_parent_batch_size=128,
        firm_batch_size=2048,output=a.run_root,device='cuda:0')
    a.review_output.mkdir(parents=True,exist_ok=True)
    atomic_write_json(a.review_output/'approval_required.json',dict(approval_binding(args,commit),approved=False,
        status='WAITING_FOR_INDEPENDENT_USER_REVIEW; NOT_EXECUTABLE_APPROVAL'))
    atomic_write_json(a.review_output/'training_plan.json',plan(coverage_config(args),firm_coverage_config(args),end_outer=15,skip_bp_head=True))
    atomic_write_json(a.review_output/'input_provenance.json',dict(source_commit=commit,
        source_job=702636,source_outer=11,source_certification='PASS',
        fixed_validation_parent_subset='FIRST_1024_CERTIFIED_ORIGINAL_IDENTITIES; NEVER_FIT_DRIVEN',
        old_h50_unmodified=True,formal_training_started=False,formal_slurm_submitted=False,
        conda_environment='DL_HL_READ_ONLY_SHARED_BINARIES; INDEPENDENT_JOB_PROCESS_AND_LEARNING_STATES'))
    environment=dict(REPO_DIR=str(ROOT),REQUIRED_COMMIT=commit,RUN_ROOT=str(a.run_root),
        SAMPLING_DIR=str(a.sampling_dir),INITIAL_CHECKPOINT=str(a.initial_checkpoint),
        SOURCE_JOURNAL_CHECKPOINT=str(a.source_journal_checkpoint),SOURCE_LATEST_POINTER=str(a.source_latest_pointer),
        SOURCE_ARTIFACT_DIR=str(a.source_artifact_dir),USER_APPROVAL_JSON='REQUIRES_NEW_USER_APPROVED_V6_FILE',RESUME='0')
    atomic_write_json(a.review_output/'planned_environment.json',environment)
    print(json.dumps(dict(status='PREPARED_PENDING_APPROVAL',source_commit=commit,formal_training_started=False,slurm_submitted=False)))


if __name__=='__main__':main()
