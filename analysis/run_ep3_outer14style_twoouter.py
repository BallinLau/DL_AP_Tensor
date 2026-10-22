"""EP3 -> Outer14-style two-Outer weights-only continuation.

New independent experiment. It cold-starts from the legacy Episode 3 weights
(``WEIGHTS_ONLY_NEW_OPTIMIZERS``) and runs exactly two new Outers
(``EP3_CONT_OUTER_1`` / ``EP3_CONT_OUTER_2``) under the reviewed Outer14
H10/H100 long-run macro replay training mechanism.

Scope is intentionally minimal and preserves every Outer14 economic equation,
Default/Exit timing, SDF formula, FC1 input definition, Firm architecture and
budget. Nothing here changes economics. The only differences from the H10/H100
runner are the initialization path (weights-only instead of full learning-state
resume) and the number of Outers (two, not four).

Q1 (confirmed): WEIGHTS_ONLY_NEW_OPTIMIZERS.
Q2 (confirmed): Outer14 default JointCoverageConfig with a LEGACY_INITIAL
reference structure and NO Outer14 learned Empirical Pool. Round one Empirical
quota is empty, so it is filled entirely by STRICT_GLOBAL h-star + economic
ledger certified Beta Coverage; round two uses this experiment's own committed
round-one Empirical Pool. All source fractions / legality / underfill are
recorded by the reused ``build_joint_bank`` audit.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.nested_equilibrium_utils import (
    atomic_write_json, atomic_torch_save, stable_json_hash, module_hash)
from analysis.finite_direct_outer4_state import StageJournal
from analysis.finite_direct_continuation_state import safe_load
from analysis.run_full_nested_finite_direct_outer4 import (
    Budget, stages, stage_id, ProductionBackend)
from analysis.run_fc1_self_consistent_macro import hashes
from analysis.run_h10_h100_longrun_macro import (
    H50Backend, coverage_config, firm_coverage_config, broad_firm_bank,
    distinct_balanced_donors, plan as h50_plan)
from analysis.h50_source_certification import (
    CONTRACT as _H50_CONTRACT, save_bank_reference, load_bank_reference)
from data.finite_direct_runtime import load_locked_sampling
from data.finite_direct_sampling import file_hash, state_hash
from data.finite_direct_joint_coverage import JointCoverageConfig
from data.finite_direct_uniform_firm_macro import (
    FirmMacroCoverageConfig, configured_macro_bank, firm_coverage_summary)
from data.finite_direct_replay_runtime import ShardStore
from data.finite_direct_longrun_replay import (
    load_pool, select_mixture, fit_edges, commit_pool)
from data.finite_direct_h100_initial import design_initial
from data.finite_direct_long_horizon_sampling import broad_firm_bank as _broad_firm_bank
from analysis.run_fc1_self_consistent_macro import simulator, select_state
from analysis.economic_config import AnalysisEconomicConfig
from data.finite_direct_joint_coverage import build_joint_bank

# ---- independent experiment identity ----
SCHEMA = 'CONTINUATION_APPROVAL_V1_EP3_OUTER14STYLE_TWO_OUTER'
EXPERIMENT_ID = 'ep3_outer14style_twoouter_20261022'
START_OUTER = 12  # internal; displayed as EP3_CONT_OUTER_1
END_OUTER = 13    # internal; displayed as EP3_CONT_OUTER_2
OUTER_LABELS = {12: 'EP3_CONT_OUTER_1', 13: 'EP3_CONT_OUTER_2'}


def plan(coverage=None, firm_coverage=None, *, end_outer=END_OUTER, skip_bp_head=True):
    if end_outer != END_OUTER:
        raise ValueError('EP3 Outer14-style experiment budget is exactly two Outers')
    firm_coverage = firm_coverage or firm_coverage_config()
    effective_firm_rows = (
        dict(train=firm_coverage.firm_macro_sampling_budget,
             validation=min(2048, firm_coverage.firm_macro_sampling_budget))
        if firm_coverage.firm_macro_coverage_enabled else dict(train=8192, validation=2048))
    execution = [dict(outer=START_OUTER, kind='baseline', indices={})]
    for o, k, i in stages(replace(Budget(), equilibrium_outer_max=(end_outer - START_OUTER + 1))):
        if o < 1:
            continue
        if k == 'cross_block':
            execution.append(dict(outer=o + START_OUTER - 1, kind='EMPIRICAL_POOL_COMMIT', indices={}))
        execution.append(dict(outer=o + START_OUTER - 1, kind=k, indices=i))
        if k == 'outer_commit':
            execution.append(dict(outer=o + START_OUTER - 1, kind='LONGRUN_MACRO_POOL_COMMIT', indices={}))
    return dict(
        version='EP3_OUTER14STYLE_TWO_OUTER_V1', experiment_id=EXPERIMENT_ID,
        continuation_contract=None, horizon=10, primary_branch=0, children=2,
        longrun=dict(paths=200, firms=200, horizon=100, branches=1, burnin=20,
                     root_consistent=100, wide_belief=100, initial_design='SCRAMBLED_SOBOL',
                     semantics='HISTORICAL_POLICY_MACRO_REPLAY', no_optimizer=True),
        initial_train_rule='LEGACY_INITIAL_REFERENCE; EMPTY_EP3_EMPIRICAL_QUOTA_FILLED_BY_BETA_COVERAGE',
        joint_coverage=asdict(coverage or JointCoverageConfig()),
        firm_macro_coverage=asdict(firm_coverage),
        effective_firm_batch_size=firm_coverage.firm_macro_batch_size if firm_coverage.firm_macro_coverage_enabled else 512,
        start_outer=START_OUTER, end_outer=end_outer, train_parents=4096, validation_parents=1024,
        firms=200, outer_labels=OUTER_LABELS, bp_head_training_skipped_by_user=skip_bp_head,
        simulation_bp_action_source='grid',
        firm_donor_sampling='DISTINCT_FIRM_TIME_BRANCH_BALANCED_OBSERVED_SUBSET',
        sdf_no_pairs='SKIP_WITHOUT_OPTIMIZER_OR_AL_MUTATION',
        initial_root=asdict(__import__('data.market_clearing', fromlist=['MarketClearingConfig']).MarketClearingConfig()),
        root_only_t0=True,
        finite_operator='finite_direct_parent_exit_before_child_v2',
        budget=asdict(Budget()), firm_rows=effective_firm_rows,
        calculated_fraction=.5, macro_pair_budget=dict(train=4096, validation=10240),
        sdf_pair_budget=dict(train=4096, validation=10240),
        fc1_mixture=dict(longrun=2048, coverage=2048, coverage_shortfall_fill=True,
                         lineage_cap_fraction=.05),
        optimizer_budget='UNCHANGED; BOUNDED_LOCKED_SUBSET_PER_DATA_STAGE',
        firm_sampling='LINEAGE_TIME_MACRO_STRATIFIED', time_strata=[[0, 2], [3, 6], [7, 9]],
        beta_empirical='TARGET_50:50; EMPTY_EP3_QUOTA_FILLED_BY_ROOT_VALID_BETA',
        recent_archive='70:30_WHEN_AVAILABLE',
        validation='FIXED_INITIAL_BANK_AND_CRN; HSTAR_RECALIBRATED_IF_MODEL_CHANGES',
        training_distribution_is_not_population=True, fc2=False,
        stages=execution,
    )


class EP3Backend(H50Backend):
    """Weights-only cold start; every Outer14 optimizer/data operator reused."""

    def __init__(self, args, locked, initial_checkpoint_path):
        args.initial_hstar_h50_adapter = True
        # Cold start (no learning snapshot): ProductionBackend loads the converted
        # weights-only combined checkpoint and builds fresh Outer14 optimizers.
        ProductionBackend.__init__(self, args, locked, learning_snapshot=None)
        # Build the canonical Firm target and hard-sync it to the online Firm.
        self.episode.refresh_firm_target()
        self.state['initialization'] = (
            'WEIGHTS_ONLY_NEW_OPTIMIZERS_EP3_FIRM_FC1_SDF_WEALTH')
        self.state['pool_sources'] = []

        # LEGACY_INITIAL reference bank: locked sampling dir (train 4096 / val 1024).
        # Keep fixed Validation identities/CRN, never resample by fit.
        for split, target in [('validation', getattr(args, 'smoke_parents', 1024))]:
            original = self.banks[split]
            indices = torch.arange(target)
            if len(original['state']['x']) < target:
                raise ValueError('insufficient fixed Validation bank')
            original = dict(original, state=select_state(original['state'], indices, torch.device('cpu')))
            original['fingerprint'] = state_hash(original['state'])
            schedule = dict(self.schedules[split], parents=self.schedules[split]['parents'][:target])
            schedule['fingerprint'] = stable_json_hash(schedule['parents'])
            original['schedule_fingerprint'] = schedule['fingerprint']
            self.banks[split] = original
            self.schedules[split] = schedule

        self.initial_train = self.banks['train']
        self.initial_schedule = self.schedules['train']
        # Explicitly NO Outer14 learned Empirical Pool.
        self.archive_templates = []
        self.state['h50_active_outer'] = START_OUTER
        self.state['source_certification'] = dict(
            initialization='WEIGHTS_ONLY_NEW_OPTIMIZERS',
            ep3_checkpoint=str(initial_checkpoint_path),
            ep3_checkpoint_sha256=file_hash(initial_checkpoint_path),
            learned_empirical_pool_imported=False,
            legacy_initial_reference=args.sampling_dir,
        )
        self._source_bank_ref = save_bank_reference(
            self.root, self.initial_train, self.initial_schedule, outer=START_OUTER - 1,
            pool_version='LEGACY_INITIAL_T0')
        self._active_bank_ref = save_bank_reference(
            self.root, self.initial_train, self.initial_schedule, outer=START_OUTER,
            pool_version='LEGACY_INITIAL_T0')
        self.coverage_cfg = coverage_config(args)
        self.firm_coverage_cfg = firm_coverage_config(args)
        if self.firm_coverage_cfg.firm_macro_coverage_enabled:
            if self.firm_coverage_cfg.firm_macro_coverage_mode != 'HYBRID':
                raise ValueError('STANDALONE uses reviewed Firm-only runner, not Outer')
            args.firm_batch_size = self.firm_coverage_cfg.firm_macro_batch_size


def approval_binding(args, commit):
    sampling = args.sampling_dir
    return dict(
        schema_name=SCHEMA, schema_version=1, source_commit=commit,
        experiment_id=EXPERIMENT_ID,
        plan_fingerprint=stable_json_hash(plan(coverage_config(args), firm_coverage_config(args),
                                                end_outer=END_OUTER, skip_bp_head=getattr(args, 'skip_bp_head', True))),
        runtime_partition=dict(parent_batch_size=getattr(args, 'parent_batch_size', 128),
                               fit_parent_batch_size=getattr(args, 'fit_parent_batch_size', 128),
                               firm_batch_size=getattr(args, 'firm_batch_size', 512)),
        firm_macro_coverage_fingerprint=stable_json_hash(asdict(firm_coverage_config(args))),
        joint_coverage_fingerprint=stable_json_hash(asdict(coverage_config(args))),
        sampling_signature_fingerprint=json.loads((sampling / 'sampling_signature.json').read_text())['fingerprint'],
        input_artifact_sha256={n: file_hash(sampling / n) for n in
                               ('sampling_signature.json', 'train_initial_bank.pt', 'validation_initial_bank.pt')},
        initial_checkpoint_sha256=file_hash(args.initial_checkpoint),
        weights_only=True,
        operator='finite_direct_parent_exit_before_child_v2',
        horizon=10,
    )


def require_approval(args, commit):
    if not args.execute_approved:
        return False
    if args.approval is None:
        raise ValueError('explicit EP3 Outer14-style approval required')
    supplied = json.loads(args.approval.read_text())
    if supplied.get('schema_version') != 1 or supplied.get('schema_name') != SCHEMA:
        raise ValueError('EP3 Outer14-style approval schema mismatch')
    if args.sampling_dir is None or args.initial_checkpoint is None:
        raise ValueError('certified input artifact paths required')
    expected = approval_binding(args, commit)
    if supplied.get('approved') is not True or any(supplied.get(k) != v for k, v in expected.items()):
        raise ValueError('EP3 Outer14-style content-bound approval mismatch')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('clean source checkout required')
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sampling-dir', type=Path, required=True)
    p.add_argument('--initial-checkpoint', type=Path, required=True)
    p.add_argument('--coverage-config', type=Path)
    p.add_argument('--firm-macro-config', type=Path)
    p.add_argument('--skip-bp-head', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--approval', type=Path)
    p.add_argument('--execute-approved', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--parent-batch-size', type=int, default=128)
    p.add_argument('--fit-parent-batch-size', type=int, default=128)
    p.add_argument('--firm-batch-size', type=int, default=512)
    args = p.parse_args(argv)

    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if not require_approval(args, commit):
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output / 'training_plan.json',
                          plan(coverage_config(args), firm_coverage_config(args),
                               end_outer=END_OUTER, skip_bp_head=args.skip_bp_head))
        return 0

    locked = load_locked_sampling(args.sampling_dir)
    if locked[2]['config']['train_parents'] != 8192 or locked[2]['config']['validation_parents'] != 2048 or locked[2]['config']['firms'] != 200:
        raise RuntimeError('formal logical data budget changed')

    backend = EP3Backend(args, locked)
    contract = approval_binding(args, commit)
    journal = StageJournal(args.output / 'transactions', contract, backend, resume=args.resume)
    atomic_write_json(args.output / 'metadata' / 'approval_v1.json', json.loads(args.approval.read_text()))

    execution = plan(coverage_config(args), firm_coverage_config(args),
                     end_outer=END_OUTER, skip_bp_head=args.skip_bp_head)['stages']
    for stage in execution:
        o, k, i = stage['outer'], stage['kind'], stage['indices']
        started = time.monotonic()
        print(json.dumps(dict(event='STAGE_BEGIN', outer=o, outer_label=OUTER_LABELS.get(o, str(o)),
                              stage=k, indices=i)), flush=True)
        result = journal.run(stage_id(o, k, i), lambda o=o, k=k, i=i: backend.execute(o, k, i))
        atomic_write_json(args.output / 'progress.json', dict(
            outer=o, outer_label=OUTER_LABELS.get(o, str(o)), stage=k, indices=i,
            status='COMMITTED', elapsed_seconds=time.monotonic() - started,
            optimizer_steps=result.get('optimizer_steps', 'NOT_AVAILABLE_SEE_OPTIMIZER_HISTORY'),
            model_hashes=result.get('hashes_after'), latest_pointer=str(journal.pointer)))
        print(json.dumps(dict(event='STAGE_COMMITTED', outer=o, stage=k, indices=i)), flush=True)
        if k in ('data', 'SDF', 'cross_block', 'outer_commit'):
            backend.write_histories(journal.results, figures=k == 'outer_commit')

    completed = [o for o in (START_OUTER, END_OUTER) if stage_id(o, 'outer_commit', {}) in journal.completed]
    atomic_write_json(args.output / 'final_summary.json', dict(
        completed_outers=completed,
        completed_outer_labels=[OUTER_LABELS[o] for o in completed],
        TRAINING_COMPLETED=completed == [START_OUTER, END_OUTER],
        ECONOMIC_EQUILIBRIUM_CONVERGED='NOT_ASSERTED',
        source_commit=commit,
        BP_HEAD_TRAINING_SKIPPED_BY_USER=args.skip_bp_head,
        SDF_skipped_stages=[k for k, v in journal.results.items() if v.get('status') == 'SDF_SKIPPED_NO_VALID_PAIRS'],
        checkpoints=[str(args.output / f'outer_{o:03d}' / 'outer_complete.pt') for o in completed],
    ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
