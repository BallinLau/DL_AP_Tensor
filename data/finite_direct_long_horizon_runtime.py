"""Bounded one-spine/two-child data store; no optimizer or altered economics.

Each batch-time shard is atomic and content checked. Committed shards can be
replayed without sampling again; one bounded state checkpoint advances a batch.
"""
from dataclasses import asdict
from pathlib import Path
import json
import math

import pandas as pd
import torch

from analysis.economic_config import AnalysisEconomicConfig
from analysis.finite_direct_continuation_state import safe_load
from analysis.nested_equilibrium_utils import atomic_torch_save, atomic_write_json, stable_json_hash
from analysis.run_fc1_self_consistent_macro import simulator, process_node, select_state, hashes
from data.finite_direct_initial_hstar import calibrate_initial, VERSION, InitialCalibrationBlocked, SOURCE_COMMIT
from data.finite_direct_runtime import capture_node, expand_after_parent_exit, cache_provenance
from data.finite_direct_sampling import file_hash, state_hash
from data.simulate_ts_parallel import _apply_entry_batched
from data.simulation_forward import forward_policy_value_for_simulation


def node_diagnostics(state,records,*,entry_attempted):
    """Reductions only; no additional policy forward or economic mutation."""
    alive=state['alive'].bool();mass=state.get('firm_mass',torch.ones_like(state['K']))
    default=alive & (state['bar_z']>=.5)
    values=torch.stack([alive.sum(1),default.sum(1),(alive & ~default).sum(1),
        (state['K']*mass*alive).sum(1),(state['K']*mass*(alive & ~default)).sum(1)],1).cpu().tolist()
    byid={r['parent_candidate_id']:r for r in records}
    event_names=('event_potential_count','event_accepted_count','event_K_entry_gross','event_reference_K','event_reference_N')
    events={name:state[name].cpu().tolist() for name in event_names if name in state}
    for j,pid in enumerate(state['transition_path_id'].cpu().tolist()):
        r=byid[pid];n,d,s,k,kpost=values[j]
        defined=all(math.isfinite(r.get(f,float('nan'))) for f in ('C_raw_t1','K_cal_t1','Hatc_cal_t1','LnK_cal_t1')) and r.get('C_raw_t1',0)>0 and r.get('K_cal_t1',0)>0
        r.update(predecision_alive_firms=int(n),hard_default_count=int(d),post_exit_alive_firms=int(s),
            K_pre_exit=k,K_post_exit=kpost,entry_attempted=entry_attempted,
            entry_candidate_count=events.get('event_potential_count',[None]*len(records))[j] if entry_attempted else None,
            entry_accepted_count=events.get('event_accepted_count',[None]*len(records))[j] if entry_attempted else None,
            K_entry_gross=events.get('event_K_entry_gross',[None]*len(records))[j] if entry_attempted else None,
            entry_reference_K=events.get('event_reference_K',[None]*len(records))[j] if entry_attempted else None,
            entry_reference_N=events.get('event_reference_N',[None]*len(records))[j] if entry_attempted else None,
            macro_defined=defined,continuation_allowed=defined)
    return records


def continuable_nodes(state,records):
    """Continuation differs from training eligibility, without changing either
    exit or entry. A defined ledger with zero post-policy survivors can still
    supply the preserved pre-exit reference to the ORIGINAL next-child entry.
    A zero-capital/undefined node cannot supply a new macro forecast.
    """
    good=[]
    for j,r in enumerate(records):
        if all(math.isfinite(r.get(k,float('nan'))) for k in
               ('C_raw_t1','K_cal_t1','Hatc_cal_t1','LnK_cal_t1')) and r['C_raw_t1']>0 and r['K_cal_t1']>0:
            good.append(j)
            state['hatc_cal'][j]=r['Hatc_cal_t1'];state['lnk_cal'][j]=r['LnK_cal_t1']
    return good


def certify_template_accounting(values,observed):
    """Join predecision realized distributions to SAME node ledger evidence."""
    by_id={r['parent_candidate_id']:r for r in observed}
    for value in values:
        t=value.get('dynamic_template')
        if t is not None:
            record=by_id[t['source_path']]
            t['post_decision_accounting_valid']=bool(record['economic_state_valid'])
            t['accounting_evidence']=dict(parent_id=record['parent_candidate_id'],
                time=t['source_time'],branch=t['source_branch'],
                C_raw=record.get('C_raw_t1'),K_cal=record.get('K_cal_t1'),
                rule='SAME_NODE_PRODUCTION_LEDGER_NOT_FUTURE_SURVIVAL')


@torch.no_grad()
def promote_child(sim, state, time):
    """Execute only the now-applicable parent financing action, no ledger redo.

    Child observation BP was explicitly NOT applicable. The same node now
    needs its actual Grid BP for the next transition, not another investment,
    default, entry, capital update or resource account.
    """
    mask = state['alive'].bool()
    path = torch.nonzero(mask, as_tuple=False)[:, 0]
    fields = [state[k][mask] for k in ('b','z','eta','i')]
    fields += [state[k][path] for k in ('x','hatcf','lnkf')]
    inputs = torch.stack(fields, dim=1)
    out = forward_policy_value_for_simulation(sim.models['policy_value'], inputs)
    action = sim._resolve_bp_action(sim=sim, firm_state=inputs, model_output=out,
        bp_head=out.bp, hatc_cal=state['hatc_cal'][path, None],
        lnk_cal=state['lnk_cal'][path, None], path_index=path,
        firm_id=state['firm_id'][mask], economic_node_id=state['economic_node_id'][path],
        t=time, branch=-1, decision_stage='parent_transition_policy',
        bp_execution_applicable=True)
    state['bp'][mask] = action.reshape(-1)
    return state


class ShardStore:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        for name,digest in self.manifest.get('initial_cache_sha256',{}).items():
            if Path(name).name!=name or file_hash(self.root/name)!=digest:
                raise ValueError('committed initial root cache corruption')

    def chunks(self):
        for name, digest in sorted(self.manifest['shards'].items()):
            if Path(name).name != name or file_hash(self.root/name) != digest:
                raise ValueError('long-horizon shard corruption')
            yield safe_load(self.root/name)

    def rows(self):
        for shard in self.chunks():
            yield pd.DataFrame(shard['rows'])

    def captures(self):
        for shard in self.chunks():
            yield from shard['captures']


@torch.inference_mode()
def generate_long_horizon_dataset(models, episode, bank, schedule, signature, output, *,
        outer, stage, horizon=50, parent_batch_size=128, root_config=None,
        capture=True, predecision_templates=False, diagnostic_observer=None,diagnostic_metadata=None):
    if horizon != 50:
        raise ValueError('H50 data contract requires exactly 50 transitions')
    if parent_batch_size < 1:
        raise ValueError('physical parent batch must be positive')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    economic = dict(economic=asdict(AnalysisEconomicConfig.from_current_config()),
                    hyperparameters=asdict(episode.hyperparams))
    from data.market_clearing import MarketClearingConfig
    root_config = root_config or MarketClearingConfig()
    provenance = cache_provenance(models, bank, schedule, signature,
                                  outer=outer, stage=stage, economic=economic)
    provenance.update(version=VERSION, horizon=horizon, primary_branch=0,split=schedule['split'],
        source_commit=SOURCE_COMMIT,runtime_source_sha256=file_hash(__file__),
        root_config=asdict(root_config), physical_parent_batch=parent_batch_size,
        capture=capture, predecision_templates=predecision_templates)
    provenance['diagnostics_version']='H50_JOINT_COVERAGE_DIAGNOSTICS_V1'
    provenance['diagnostic_metadata']=diagnostic_metadata or {}
    provenance=json.loads(json.dumps(provenance))
    identity_path = output/'provenance.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != provenance:
        raise ValueError('long-horizon model/CRN/horizon/root provenance mismatch')
    atomic_write_json(identity_path, provenance)
    before = hashes(models); manifest = {}; root_audits = [];initial_caches={}
    n = len(bank['state']['x']); all_ids = bank['state']['transition_path_id'].tolist()
    source_specs={r['parent_id']:r for r in schedule['parents']}
    for start in range(0, n, parent_batch_size):
        stop = min(n, start+parent_batch_size)
        sim = simulator(models, episode, stop-start, schedule['seed'])
        if sim.transition_rng_mode != 'stable_firm_identity' or sim.entry_rng_mode != 'stable_economic_identity':
            raise ValueError('long-horizon requires stable identity CRN')
        initial = select_state(bank['state'], slice(start, stop), episode.device)
        root_cache=output/f'initial_{start:06d}.pt'
        if (output/f'batch_{start:06d}_time_000.pt').exists() and not root_cache.exists():
            raise ValueError('committed rollout lost initial root cache; refusing repeated solve')
        try:
            state, parents, good, audit = calibrate_initial(sim, initial, config=root_config,
                outer=outer, split=schedule['split'], economic=economic,
                crn=provenance['crn_hash'], cache=root_cache)
        except InitialCalibrationBlocked as exc:
            atomic_write_json(output/f'initial_failure_{start:06d}.json',dict(error=str(exc),audit=exc.audit))
            raise
        root_audits.extend(audit)
        node_diagnostics(state,parents,entry_attempted=False)
        if diagnostic_observer is not None:
            diagnostic_observer('INITIAL_PROCESSED',sim,state,parents,0,-1)
        if root_cache.exists():initial_caches[root_cache.name]=file_hash(root_cache)
        for time in range(horizon):
            target = output/f'batch_{start:06d}_time_{time:03d}.pt'
            digest = target.with_suffix('.sha256.json')
            if target.exists() and not digest.exists():
                # Atomic torch save already renamed a complete file, but the
                # sidecar publication was interrupted. Recover only after full
                # safe deserialization + exact provenance/coverage validation.
                unpublished=safe_load(target)
                if unpublished['provenance']!=provenance or len(unpublished['rows'])!=2*(stop-start):
                    raise ValueError('invalid unpublished rollout shard')
                atomic_write_json(digest,dict(sha256=file_hash(target)))
            if target.exists() or digest.exists():
                if not (target.exists() and digest.exists()):
                    # Unpublished file is not a committed shard. Refuse rather
                    # than overwrite; recovery is explicit and preserves it.
                    raise ValueError('unpublished long-horizon shard requires review')
                expected = json.loads(digest.read_text())
                if file_hash(target) != expected['sha256']:
                    raise ValueError('long-horizon committed shard corruption')
                saved = safe_load(target)
                if saved['provenance'] != provenance:
                    raise ValueError('stale long-horizon committed shard')
                state = select_state(saved['next_state'], slice(None), episode.device) if saved['next_state'] is not None else None
                parents, good = saved['next_parents'], saved['next_good']
            else:
                rows, captures, templates = [], [], []
                parent_records=parents
                attempted_parent_ids=[]
                next_state = None; next_parents = []; next_good = []
                if state is not None and good:
                    if time and capture:
                        captures.append(capture_node(state, parents, split=schedule['split'],
                            outer=outer, time=time, branch=0, model_hashes=before))
                    elif time == 0 and capture:
                        captures.append(capture_node(state, parents, split=schedule['split'],
                            outer=outer, time=0, branch=-1, model_hashes=before))
                    state = select_state(state, torch.tensor(good, device=episode.device))
                    attempted_parent_ids=state['transition_path_id'].cpu().tolist()
                    if time:
                        state = promote_child(sim, state, time)
                        if diagnostic_observer is not None:
                            diagnostic_observer('NEXT_PARENT_BP_STAGE',sim,state,parents,time,0)
                    pmap = {r['parent_candidate_id']:r for r in parents}
                    children = expand_after_parent_exit(sim, state, child_t=time+1)
                    for branch, child in enumerate(children):
                        if diagnostic_observer is not None:
                            diagnostic_observer('POST_EXIT',sim,child,[],time+1,branch)
                        child = _apply_entry_batched(sim, child, max(20,int(sim.group_size*.1)))
                        if diagnostic_observer is not None:
                            diagnostic_observer('POST_ENTRY_PRE_DECISION',sim,child,[],time+1,branch)
                        values=[]
                        if predecision_templates and schedule['split']=='train':
                            from data.finite_direct_dynamic_empirical import capture_templates
                            values = capture_templates(child, split='train', outer=outer,
                                branch=branch, model_hashes=before)
                            for value in values:
                                item=value.get('dynamic_template',value.get('template_failure'))
                                item['source_time']=time+1
                                if 'template_id' in item:
                                    item['template_id'] += f':time{time+1}'
                                value['source_time']=time+1
                            templates.extend(values)
                        if bool(child['alive'].any()):
                            child, observed, valid = process_node(sim, child, time+1, branch, entry=False)
                        else:
                            observed=[dict(parent_candidate_id=int(pid),t=time+1,child_branch=branch,
                                economic_state_valid=False,failure_reason='ECONOMICALLY_EXTINCT_AFTER_ENTRY',
                                entry_executed=True,x_t1=float(child['x'][j]),
                                Hatcf_t1=float(child['hatcf'][j]),LnKF_t1=float(child['lnkf'][j]),
                                C_raw_t1=float('nan'),K_cal_t1=0.,
                                Hatc_cal_t1=float('nan'),LnK_cal_t1=float('nan'))
                                for j,pid in enumerate(child['transition_path_id'].tolist())]
                        valid=continuable_nodes(child,observed)
                        node_diagnostics(child,observed,entry_attempted=True)
                        certify_template_accounting(values,observed)
                        if diagnostic_observer is not None:
                            diagnostic_observer('POST_DECISION',sim,child,observed,time+1,branch)
                        if capture and branch==1 and time+1<horizon and bool(child['alive'].any()):
                            captures.append(capture_node(child,observed,split=schedule['split'],
                                outer=outer,time=time+1,branch=branch,model_hashes=before))
                        for original_row in observed:
                            row=dict(original_row)
                            parent = pmap[row['parent_candidate_id']]
                            row.update(split=schedule['split'], economic_time=time,
                                node_attempted=True,
                                x_t=parent['x_t1'], Hatc_cal_t=parent['Hatc_cal_t1'],
                                LnK_cal_t=parent['LnK_cal_t1'], Hatcf_t=parent['Hatcf_t1'],
                                LnKF_t=parent['LnKF_t1'])
                            row['node_economic_state_valid']=original_row['economic_state_valid']
                            row['node_failure_reason']=original_row['failure_reason']
                            row['prediction_input_kind']='TRUE_PARENT_CALCULATED'
                            if not parent['economic_state_valid']:
                                row['economic_state_valid']=False
                                row['failure_reason']='PARENT_'+parent['failure_reason']
                            row['continuation_status']=('DEFINED_LEDGER_NEXT_ENTRY_PERMITTED'
                                if original_row.get('C_raw_t1',0)>0 and original_row.get('K_cal_t1',0)>0
                                else 'ECONOMICALLY_UNDEFINED')
                            rows.append(row)
                        if branch == 0:
                            next_parents=observed;next_good=valid
                            if valid:next_state=child
                    del children
                # Preserve complete attempted path-time coverage. Undefined
                # suffixes have no imputed numeric targets and are never fit.
                seen={(r['parent_candidate_id'],r['child_branch']) for r in rows}
                for pid in all_ids[start:stop]:
                    for branch in (0,1):
                        if (pid,branch) not in seen:
                            rows.append(dict(split=schedule['split'],parent_candidate_id=pid,
                                t=time+1, economic_time=time, child_branch=branch,
                                economic_state_valid=False,
                                node_attempted=False,
                                failure_reason='ANCESTOR_UNDEFINED_OR_EXTINCT'))
                for row in rows:
                    spec=source_specs[row['parent_candidate_id']]
                    row.update(sampling_source=spec.get('sampling_source','UNKNOWN'),
                        source_type=spec.get('source_type','LEGACY_'+spec.get('sampling_source','UNKNOWN')),
                        b_distribution_family=spec.get('coverage_family','LEGACY_'+spec.get('sampling_source','UNKNOWN')),
                        source_lineage=spec.get('source_lineage',row['parent_candidate_id']),
                        source_outer=spec.get('source_outer',-1),source_snapshot_time=spec.get('source_snapshot_time',0),
                        source_pool_fingerprint=spec.get('source_pool_fingerprint','LEGACY_INITIAL'),
                        sampling_weight=spec.get('sampling_weight',1.),reuse_count=spec.get('reuse_count',1))
                saved=dict(provenance=provenance, rows=rows, captures=captures,parent_records=parent_records,
                    attempted_parent_ids=attempted_parent_ids,
                    templates=templates, next_state=None if next_state is None else
                    select_state(next_state,slice(None),torch.device('cpu')),
                    next_parents=next_parents,next_good=next_good)
                atomic_torch_save(target,saved)
                atomic_write_json(digest,dict(sha256=file_hash(target)))
                state,parents,good=next_state,next_parents,next_good
            manifest[target.name]=file_hash(target)
    if hashes(models)!=before:
        raise RuntimeError('long-horizon simulation mutated frozen models')
    pd.DataFrame(root_audits).to_csv(output/'initial_hstar_audit.csv',index=False)
    result=dict(provenance=provenance,shards=manifest,scheduled_initial_parents=n,
        initial_cache_sha256=initial_caches,
        scheduled_parent_times=n*horizon,scheduled_child_rows=2*n*horizon,
        real_root_feasibility='PASS',training_performed=False)
    atomic_write_json(output/'manifest.json',result)
    return ShardStore(output)
