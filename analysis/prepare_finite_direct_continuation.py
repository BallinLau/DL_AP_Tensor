#!/usr/bin/env python3
"""Non-optimizing preparation only; never issues an approval or submits a job."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import shutil
import subprocess
import sys
import zipfile
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.finite_direct_continuation_state import (RUN_ID,SOURCE_SHA,certify_source,
    extract_learning_state,state_digest)
from analysis.run_finite_direct_large_continuation import training_plan
from analysis.nested_equilibrium_utils import atomic_torch_save,atomic_write_json,stable_json_hash
from data.finite_direct_continuation_sampling import prepare_sampling,CONFIG
from data.finite_direct_sampling import file_hash

TEMPLATE_SHA='f616821c7407230eef850a255ea1205a1df47d3b3e659d1b2820cfd810ebf1d3'


def prepare(args):
    payload,audit=certify_source(args.source_run)
    learning=extract_learning_state(payload)
    args.output.mkdir(parents=True,exist_ok=True)
    sampling=args.output/'continuation_sampling'
    locked=prepare_sampling(sampling,args.template_bank,TEMPLATE_SHA)
    from analysis.finite_direct_dynamic_approval import prepare_initial_pool
    prepare_initial_pool(sampling)
    sig=locked[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    initial=dict(format='finite_direct_cross_run_continuation_v1',source_job_id=701387,
        source_outer=3,source_stage='outer_003/outer_commit',learning=learning,
        learning_sha256=state_digest(learning),new_sampling_signature=sig['fingerprint'])
    path=args.output/'continuation_initial_checkpoint.pt'
    if path.exists():
        from analysis.finite_direct_continuation_state import safe_load
        old=safe_load(path)
        if state_digest(old)!=state_digest(initial):raise ValueError('existing initial state differs; no overwrite')
    else: atomic_torch_save(path,initial)
    audit.update(learning_sha256=state_digest(learning),new_sampling_signature=sig['fingerprint'],
        continuation_initial_checkpoint_sha256=file_hash(path),new_training_budget=training_plan()['budget'],
        new_code_commit=commit,source_run_verified_on_thu=True,
        historical_operator='finite_direct_production_ledger_v1',
        new_operator=training_plan()['operator'],historical_convergence_evidence_for_new_operator=False,
        source_run='/home/fit/zhuyingz/WORK/LiuHao/cachedir/full_nested_finite_direct_outer4_20261020_160a4d1_r1',
        local_readonly_mirror=str(args.source_run.resolve()),training_performed=False)
    audit['source_component_hashes']=audit['model_hashes']
    audit['source_optimizer_state_hashes']={k:v['sha256'] for k,v in audit['optimizers'].items()}
    audit['source_journal_checkpoint_sha256']=SOURCE_SHA
    atomic_write_json(args.output/'continuation_source_manifest.json',audit)
    atomic_write_json(args.output/'continuation_training_plan.json',training_plan())
    review=args.output/'review';review.mkdir(exist_ok=True)
    atomic_write_json(review/'source_checkpoint_audit.json',audit)
    atomic_write_json(review/'source_optimizer_al_audit.json',{k:v for k,v in audit.items()
        if k in ('optimizers','source_al_state','source_al_hash','scheduler_hash','source_sdf_epoch',
                 'source_sdf_optimizer_steps','episode_runtime_hash','source_rng_hash','source_fresh_hash')})
    pd.DataFrame([dict(check=k,result='PASS',value=json.dumps(v,sort_keys=True)) for k,v in audit.items()
        if k in ('model_hashes','optimizers','source_al_state','source_terminal_checkpoint_sha256',
                 'outer_complete_learning_state_parity')]).to_csv(args.output/'continuation_source_audit.csv',index=False)
    coverage=json.loads((sampling/'sampling_coverage_summary.json').read_text())
    pd.DataFrame([dict(split=s,**v) for s,v in coverage['splits'].items()]).to_csv(args.output/'sampling_coverage.csv',index=False)
    shutil.copyfile(sampling/'empirical_template_manifest.csv',args.output/'sampling_template_manifest.csv')
    historical=args.output/'historical_baseline';historical.mkdir(exist_ok=True)
    if args.historical_zip:
        with zipfile.ZipFile(args.historical_zip) as archive:
            for member in archive.namelist():
                parts=Path(member).parts[1:]
                if not parts:continue
                rel=Path(*parts)
                # Never duplicate the OLD approved=true authorization into the
                # new prepared artifacts, even as a historical convenience.
                if 'approval' in rel.name.lower():continue
                if (len(parts)==1 and (rel.suffix=='.csv' or rel.name=='final_summary.json')) or parts[0]=='metadata':
                    if '..' in parts: raise ValueError('unsafe archive path')
                    target=historical/rel;target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(archive.read(member))
    atomic_write_json(historical/'historical_contract.json',dict(job=701387,source_commit=payload['contract']['source_commit'],
        sampling_signature=payload['contract']['sampling_signature'],source_checkpoint_sha256=SOURCE_SHA,
        representation='FINITE_FIRM_DIRECT_AGGREGATION',validation_bank_changed=True,
        comparison='Historical 0-3 are separate; only new-bank baseline3 vs4-11 is same-CRN comparable'))
    audit['historical_artifacts']={p.relative_to(historical).as_posix():file_hash(p)
        for p in sorted(historical.rglob('*')) if p.is_file()}
    atomic_write_json(args.output/'continuation_source_manifest.json',audit)
    atomic_write_json(review/'source_checkpoint_audit.json',audit)
    audit['sampling_coverage']=coverage
    print(json.dumps(dict(status='PASS',output=str(args.output),source_sha256=SOURCE_SHA,
        initial_sha256=file_hash(path),sampling_signature=sig['fingerprint'],coverage=coverage['splits'],
        training_performed=False),indent=2))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--template-bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--historical-zip',type=Path)
    return p.parse_args(argv)


if __name__=='__main__':prepare(parse_args())
