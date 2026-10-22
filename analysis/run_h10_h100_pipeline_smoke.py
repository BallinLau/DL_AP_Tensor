"""Explicitly bounded GPU integration test, never the formal four-Outer job.

Two tiny cycles exercise actual P/Q, production FC1/SDF, and next-Outer pool
activation through StageJournal. No production approval artifact is created.
"""
import argparse,json,sys,time,subprocess
from pathlib import Path
from dataclasses import asdict,replace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.run_h10_h100_longrun_macro import H50Backend
from analysis.run_full_nested_finite_direct_outer4 import Budget,stage_id
from analysis.finite_direct_outer4_state import StageJournal
from analysis.finite_direct_continuation_state import safe_load
from analysis.nested_equilibrium_utils import atomic_write_json
from data.finite_direct_joint_coverage import JointCoverageConfig
from data.finite_direct_uniform_firm_macro import FirmMacroCoverageConfig
from data.finite_direct_runtime import load_locked_sampling


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('sampling-dir','initial-checkpoint','source-journal-checkpoint','source-latest-pointer','source-artifact-dir','output'):
        p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not str(a.output).endswith('_pipeline_smoke'):raise ValueError('explicit smoke-only output suffix required')
    a.output.mkdir(parents=True,exist_ok=True)
    a.device='cuda:0';a.parent_batch_size=8;a.fit_parent_batch_size=8;a.firm_batch_size=64
    a.smoke_parents=16;a.smoke_longrun_paths=2;a.skip_bp_head=True
    a.coverage_config=a.output/'smoke_coverage.json';a.firm_macro_config=a.output/'smoke_firm.json'
    atomic_write_json(a.coverage_config,asdict(JointCoverageConfig(max_candidates=64,candidate_batch_size=8)))
    atomic_write_json(a.firm_macro_config,asdict(FirmMacroCoverageConfig(firm_macro_coverage_enabled=True,
        firm_macro_coverage_fraction=.5,firm_macro_batch_size=64,firm_macro_sampling_budget=128)))
    backend=H50Backend(a,load_locked_sampling(a.sampling_dir),safe_load(a.initial_checkpoint))
    backend.budget=replace(Budget(),sdf_epochs_per_chunk=1)
    backend.solver_args.p_adam_steps=1;backend.solver_args.p_lbfgs_max_iter=1
    backend.solver_args.p_lbfgs_chunk_iterations=1;backend.solver_args.q_fit_steps=1
    contract=dict(scope='H10_H100_PIPELINE_SMOKE_NOT_FORMAL',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    journal=StageJournal(a.output/'transactions',contract,backend,resume=a.resume)
    started=time.monotonic()
    sequence=[('outer_start',{}),('firm_bank',{}),('P',dict(inner=0,policy=0)),('Q',dict(inner=0,picard=0)),
        ('BP',dict(inner=0)),('data',dict(version=0)),('FC1',dict(macro=0,epoch=1)),
        ('data',dict(version=1)),('FC1',dict(macro=1,epoch=1)),('data',dict(version=2)),
        ('SDF',dict(chunk=0)),('post_sdf_validation',{}),('EMPIRICAL_POOL_COMMIT',{}),
        ('cross_block',{}),('outer_commit',{}),('LONGRUN_MACRO_POOL_COMMIT',{})]
    for outer in (12,13):
        for kind,indices in sequence:
            print(json.dumps(dict(event='SMOKE_BEGIN',outer=outer,stage=kind)),flush=True)
            result=journal.run(stage_id(outer,kind,indices),lambda:backend.execute(outer,kind,indices))
            print(json.dumps(dict(event='SMOKE_COMMIT',outer=outer,stage=kind,result=result),default=str),flush=True)
    steps={kind:sum(v.get('optimizer_steps',int(v.get('optimizer_executed',False))) for v in journal.results.values() if v['stage']==kind)
        for kind in ('P','Q','FC1','SDF')}
    passed=all(steps[k]>0 for k in steps)
    atomic_write_json(a.output/'summary.json',dict(status='PASS' if passed else 'INCOMPLETE_OPTIMIZER_COVERAGE',
        smoke_only=True,formal_training_started=False,optimizer_steps=steps,completed=journal.completed,
        h100_paths_per_outer=4,elapsed_seconds=time.monotonic()-started))
    if not passed:raise RuntimeError('smoke optimizer coverage incomplete; do not approve formal run')


if __name__=='__main__':main()
