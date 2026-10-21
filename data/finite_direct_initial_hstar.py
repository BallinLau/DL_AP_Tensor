"""Initial-only STRICT_GLOBAL adapter for finite-direct Exit-v2.

No new root algorithm, no training, no approximation fallback.  The initial
node has no entry; exit is committed by expand_after_parent_exit, NOT here.
"""
from dataclasses import asdict
from pathlib import Path
import time
import subprocess

import torch

from analysis.finite_direct_continuation_state import safe_load, state_digest
from analysis.nested_equilibrium_utils import atomic_torch_save, stable_json_hash
from analysis.run_fc1_self_consistent_macro import hashes, process_node, select_state
from data.finite_direct_runtime import OPERATOR
from data.finite_direct_sampling import file_hash, state_hash
from data.market_clearing import (MarketClearingConfig, MarketClearingController,
                                  MarketClearingError, STRICT_GLOBAL,
                                  empty_entry_candidates)

VERSION = 'INITIAL_HSTAR_H50_BROAD_MACRO_V1'
SOURCE_COMMIT=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip()


class InitialCalibrationBlocked(RuntimeError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def identity(models, state, *, config, economic, crn):
    return dict(version=VERSION, operator=OPERATOR, models=hashes(models),
                bank=state_hash(state), root=asdict(config), economic=economic,
                crn=crn, entry_at_t0=False,
                adapter_source_sha256=file_hash(__file__),
                source_commit=SOURCE_COMMIT,
                production_source_sha256={name:file_hash(Path(__file__).parent/name) for name in
                    ('market_clearing.py','simulate_ts_parallel.py','simulate_ts.py','entry.py','simulation_forward.py')},
                timing='DEFINED_LEDGER_CONTINUES_TO_ORIGINAL_EXIT_V2_AND_CHILD_ENTRY')


@torch.no_grad()
def calibrate_initial(sim, state, *, config=None, outer, split, cache=None,
                      economic, crn, economic_time=0):
    """Return a processed t0 node and audit; never replace a failed parent.

    A cached node owns detached CPU tensors and must pass its content digest.
    Cache identity includes original beliefs, every network and solver setting.
    A cache hit does not process the node again. No old controller or boundary
    integration cache can be accepted by this adapter.
    """
    if economic_time != 0:
        raise ValueError('h-star is permitted only at t=0')
    config = config or MarketClearingConfig()
    config.validate()
    if config.solver_mode != STRICT_GLOBAL:
        raise ValueError('initial calibration requires STRICT_GLOBAL')
    if getattr(sim, 'aggregation_mode', None) != 'legacy_mixed_replay':
        raise ValueError('initial calibration requires finite-direct accounting')
    if sim.market_clearing_controller is not None:
        raise ValueError('per-period clearing must remain disabled')
    original_hash = state_hash(state)
    provenance = identity(sim.models, state, config=config, economic=economic, crn=crn)
    cache = None if cache is None else Path(cache)
    if cache is not None and cache.exists():
        saved = safe_load(cache)
        if saved['identity'] != provenance:
            raise ValueError('initial root cache identity mismatch')
        if saved['state_hash'] != state_hash(saved['state']):
            raise ValueError('initial root cache corruption')
        if saved['payload_digest'] != state_digest({k:v for k,v in saved.items() if k!='payload_digest'}):
            raise ValueError('initial root record cache corruption')
        return select_state(saved['state'], slice(None), sim.device), saved['records'], saved['valid'], saved['audit']
    s = select_state(state, slice(None), sim.device)
    capital = (s['K'].double() * s['alive'].double() *
               s.get('firm_mass', torch.ones_like(s['K'])).double()).sum(1)
    if not bool((torch.isfinite(capital) & (capital > 0)).all()):
        raise InitialCalibrationBlocked('undefined initial capital', [])
    s['lnkf'] = capital.log().to(s['K'].dtype)
    belief = s['hatcf'].clone()
    controller = MarketClearingController(config)
    entries = empty_entry_candidates(s)
    started = time.monotonic()
    try:
        # Reuse the unmodified production root algorithm. No entry is created
        # at t0; Exit-v2 remains exclusively in the inter-period transition.
        s = controller.clear_state(sim, s, entries, t=0, branch=-1)
    except MarketClearingError as exc:
        for failure in controller.failure_records:
            failure['parent_id']=int(state['transition_path_id'][failure['path']])
        audit = dict(identity=provenance, root_records=controller.root_records,
                     failures=controller.failure_records,
                     operator_evaluations=controller.operator_evaluations,
                     wallclock_seconds=time.monotonic()-started,
                     source_snapshot_unchanged=original_hash == state_hash(state),
                     committed=False)
        raise InitialCalibrationBlocked(str(exc), audit) from exc
    selected = controller.evaluate(sim, s, entries, s['hatcf'][:, None])
    # Independent real production ledger AND actual parent Grid BP execution.
    s, records, valid = process_node(sim, s, 0, -1, entry=False)
    audit = []
    roots = {r['path']: r for r in controller.root_records}
    for j, row in enumerate(records):
        c, k = row['C_raw_t1'], row['K_cal_t1']
        candidate_c = float(selected['C_raw'][j, 0])
        candidate_k = float(selected['K_total'][j, 0])
        # FP32 ledger accumulation parity; root acceptance itself is unchanged.
        parity = (abs(c-candidate_c) <= 2e-6*max(1., abs(c)) and
                  abs(k-candidate_k) <= 2e-6*max(1., abs(k)))
        lnk_parity = abs(float(s['lnkf'][j])-row['LnK_cal_t1']) <= 2e-6
        strict = row['normalized_G_at_prediction'] <= config.residual_tolerance
        pid = row['parent_candidate_id']
        audit.append(dict(outer=outer, split=split, parent_id=pid,
            initial_lnk=float(capital[j].log()), initial_h_belief=float(belief[j]),
            selected_hstar=float(s['hatcf'][j]), root_found=True,
            root_count=roots[pid]['root_count'], root_status='ROOT_FOUND',
            normalized_residual=row['normalized_G_at_prediction'], G_raw=row['G_at_prediction'],
            C_raw=c, K_ledger=k, resource_feasible=c>0 and k>0,
            firm_default_fraction=row['default_mass']/max(row['firm_mass'], 1e-30),
            firm_investment_fraction=row['investment_mass']/max(row['firm_mass'], 1e-30),
            entry_candidate_count=0, entry_accepted_count=0,
            root_operator_evaluations=int(s['market_root_operator_evaluations'][j]),
            batch_operator_evaluations=controller.operator_evaluations,
            batch_network_forward_calls=controller.network_forward_calls,
            root_wallclock_seconds=time.monotonic()-started,
            candidate_committed_ledger_parity=parity, lnk_parity=lnk_parity,
            all_valid_roots=roots[pid]['all_valid_roots']))
        # Zero post-decision survivors is not by itself an undefined current
        # resource account. Exit-v2 preserves pre-exit entry reference; next
        # child entry must be attempted before declaring permanent extinction.
        if not (parity and lnk_parity and strict and c>0 and k>0):
            raise InitialCalibrationBlocked('initial root production replay gate failed', audit)
    valid=[j for j,r in enumerate(records) if r['C_raw_t1']>0 and r['K_cal_t1']>0
           and torch.isfinite(torch.tensor([r['Hatc_cal_t1'],r['LnK_cal_t1']])).all()]
    for j in valid:
        s['hatc_cal'][j]=records[j]['Hatc_cal_t1'];s['lnk_cal'][j]=records[j]['LnK_cal_t1']
    if state_hash(state) != original_hash:
        raise RuntimeError('candidate evaluation mutated initial bank')
    if hashes(sim.models) != provenance['models']:
        raise RuntimeError('initial calibration mutated model')
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cpu = select_state(s, slice(None), torch.device('cpu'))
        payload=dict(identity=provenance,state=cpu,state_hash=state_hash(cpu),
                     records=records,valid=valid,audit=audit)
        payload['payload_digest']=state_digest(payload)
        atomic_torch_save(cache,payload)
    return s, records, valid, audit
