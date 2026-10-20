"""New-run sampling namespace; unchanged finite-direct economic initialization."""
from dataclasses import asdict
from pathlib import Path
import json
import pandas as pd
from data.finite_direct_sampling import (SamplingConfig,build_or_load_sampling,validate_template_splits,file_hash)
from data.finite_direct_runtime import load_locked_sampling
from analysis.finite_direct_continuation_state import RUN_ID,SOURCE_SHA,safe_load
from analysis.nested_equilibrium_utils import atomic_write_json,stable_json_hash

CONFIG=SamplingConfig(train_parents=8192,validation_parents=2048,seed=202610204)


def prepare_sampling(output, template_path, expected_template_sha):
    output=Path(output)
    if file_hash(template_path)!=expected_template_sha: raise ValueError('unreviewed empirical template file')
    bank=safe_load(template_path); counts=validate_template_splits(bank)
    if counts['train_templates']!=4096 or counts['validation_templates']!=1024:
        raise ValueError('full original empirical templates unavailable')
    # Original v2 source-kind string is retained, not rewritten to pretend
    # on-policy data. Individual templates explicitly identify source_time=0.
    if any(t['source_time']!=0 for t in bank['templates']): raise ValueError('not the reviewed t0 template source')
    if (output/'sampling_signature.json').exists():
        saved=json.loads((output/'sampling_signature.json').read_text())
        if saved.get('run_identity')!=RUN_ID or saved.get('source_checkpoint_sha256')!=SOURCE_SHA:
            raise ValueError('wrong continuation sampling namespace')
        result=load_locked_sampling(output)
        if saved['config']!=asdict(CONFIG): raise ValueError('wrong logical sampling capacity')
        return result
    manifest=pd.DataFrame([{k:v for k,v in t.items() if k not in ('b_sorted','b_by_capital_quartile')}
                           for t in bank['templates']])
    build_or_load_sampling(output,CONFIG,bank,manifest)
    path=output/'sampling_signature.json'; sig=json.loads(path.read_text())
    sig.update(base_sampling_fingerprint=sig['fingerprint'],run_identity=RUN_ID,
        source_checkpoint_sha256=SOURCE_SHA,empirical_semantics='RECORDED_INITIAL_PARENT_DISTRIBUTION',
        identity_contract='run_id + split + parent_id + economic_time + branch_id')
    sig['fingerprint']=stable_json_hash({k:v for k,v in sig.items() if k not in ('fingerprint','artifact_sha256')})
    atomic_write_json(path,sig)
    result=load_locked_sampling(output)
    for split in ('train','validation'):
        rows=result[1][split]['parents']; ids=[r['empirical_template_id'] for r in rows if r['sampling_source']=='EMPIRICAL']
        if len(set(ids))!=len(ids): raise ValueError('unexpected empirical template reuse')
    return result
