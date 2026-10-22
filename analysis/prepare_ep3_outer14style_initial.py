"""Convert legacy EP3 weights-only checkpoint to Outer14 cold-start combined checkpoint.

Review/preparation only. No optimizer, no Slurm, no economic simulation.

Q1 decision: WEIGHTS_ONLY_NEW_OPTIMIZERS
  - strict-load Firm (policy_value), FC1/SDF/Wealth (sdf_fc1) into current Outer14
    architectures;
  - verify ep3 firm_target is loadable and record its hash (do NOT import it over
    online Firm; the Outer14 convention is refresh_firm_target hard-sync online->target);
  - do NOT import old optimizer/scheduler/RNG/AL state;
  - new optimizers/schedulers are rebuilt from Outer14 config; AL is reset once
    at experiment init (lambda=0, rho=2), and inherited within later chunks/Outers.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_utils import build_models, build_policy_value_from_checkpoint_spec
from analysis.nested_equilibrium_utils import atomic_torch_save, atomic_write_json, tensor_hash
from analysis.run_fc1_self_consistent_macro import hashes
from data.finite_direct_sampling import file_hash

FORMAT = 'finite_direct_outer4_initial_weights_v1'


def build_combined(ep3_path: Path, *, device: str = 'cpu'):
    cp = torch.load(ep3_path, map_location='cpu', weights_only=False)
    if not isinstance(cp, dict) or 'models' not in cp:
        raise ValueError('ep3 checkpoint is not a combined model checkpoint')
    spec = cp.get('policy_value_model_spec')
    if not isinstance(spec, dict):
        raise ValueError('ep3 checkpoint missing policy_value_model_spec')
    q_param = str(spec.get('q_parameterization', 'b_times_unit')).lower()
    if q_param != 'direct':
        raise ValueError(f'ep3 q_parameterization must be direct, got {q_param!r}')

    # Build fresh Outer14 architectures and strict-load the online weights.
    models = build_models(torch.device('cpu'))
    models['policy_value'].load_state_dict(cp['models']['policy_value'], strict=True)
    models['sdf_fc1'].load_state_dict(cp['models']['sdf_fc1'], strict=True)
    for m in models.values():
        m.eval(); m.requires_grad_(False)

    # ep3 firm_target verification: loadable strict into the same spec.
    firm_target = build_policy_value_from_checkpoint_spec(
        cp, value_scale_mode=None, value_scale_log_max=None).to('cpu')
    firm_target.load_state_dict(cp['models']['firm_target'], strict=True)
    firm_target.eval(); firm_target.requires_grad_(False)
    firm_target_hash = tensor_hash(sorted(firm_target.state_dict().items()))

    source_manifest = dict(
        format=FORMAT,
        ep3_checkpoint=str(ep3_path),
        ep3_checkpoint_sha256=file_hash(ep3_path),
        ep3_git_commit=cp.get('git_commit'),
        ep3_episode=cp.get('episode'),
        ep3_warmstart=cp.get('warmstart'),
        q_parameterization=q_param,
        value_parameterization=cp.get('value_parameterization'),
        policy_value_model_spec=spec,
        firm_target_verified_loadable=True,
        firm_target_state_hash=firm_target_hash,
        firm_target_semantics=(
            'ep3 firm_target verified loadable; live target re-derived from online '
            'Firm via refresh_firm_target(), never imported stale.'
        ),
        optimizer_state_restored=False,
        rng_state_restored=False,
        simulation_state_restored=False,
        training_data_restored=False,
        al_reset_once_at_init=dict(lambda_mean_low=0.0, lambda_mean_high=0.0,
                                   lambda_var_high=0.0, rho=2.0),
        loaded_model_keys=['policy_value', 'sdf_fc1'],
        hashes=hashes(models),
        initialization='WEIGHTS_ONLY_NEW_OPTIMIZERS',
    )
    combined = dict(
        format=FORMAT,
        models={
            'policy_value': models['policy_value'].state_dict(),
            'sdf_fc1': models['sdf_fc1'].state_dict(),
            'firm_target': firm_target.state_dict(),
        },
        source_manifest=source_manifest,
        training_performed=False,
    )
    return combined, source_manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ep3-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)

    combined, manifest = build_combined(args.ep3_checkpoint)
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(out, combined)
    manifest['combined_sha256'] = file_hash(out)
    atomic_write_json(out.with_suffix('.json'), manifest)
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
