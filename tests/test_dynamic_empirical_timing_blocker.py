"""Non-optimization evidence tests; a PASS does NOT make dynamic sampling ready."""
from copy import deepcopy
import ast
import inspect
import torch
from analysis.audit_dynamic_empirical_timing import fixture, reproduce
from data.simulate_ts_parallel import _expand_branches_batched, _apply_exit_batched
from data.finite_direct_runtime import generate_dataset, capture_node


def test_realized_b_uses_parent_eta_and_not_unexecuted_bp():
    result=reproduce()
    assert result['realized_b_eta_t_parity']
    for row in result['branches']:
        assert row['child_b'][0][1] == torch.tensor(.4).item()  # eta_t=0, not bp=.9


def test_production_expansion_reproduces_previous_default_leak():
    for row in reproduce()['branches']:
        assert row['parent_defaulted_ids']==[10]
        assert row['previous_default_still_alive_ids']==[10]
        assert row['child_alive_ids']==[10,11]


def test_existing_exit_on_copy_removes_previous_default_without_changing_b():
    sim,parent=fixture();child=_expand_branches_batched(sim,deepcopy(parent))[0]
    copy=deepcopy(child);before=child['b'].clone();_apply_exit_batched(copy)
    assert copy['alive'].tolist()==[[False,True,False]]
    assert child['alive'].tolist()==[[True,True,False]]
    torch.testing.assert_close(before,copy['b'])


def test_stable_crn_replay_and_previous_dead_identity_stays_dead():
    a=reproduce();b=reproduce();assert a==b
    assert all(12 not in r['child_alive_ids'] for r in a['branches'])


def test_adapter_now_uses_corrected_transition_but_old_capture_remains_post_decision():
    tree=ast.parse(inspect.getsource(generate_dataset.__wrapped__))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    assert not any(n.func.id=='_apply_exit_batched' for n in calls)
    assert any(n.func.id=='expand_after_parent_exit' for n in calls)
    child_process=[n for n in calls if n.func.id=='process_node' and len(n.args)>3 and isinstance(n.args[2],ast.Constant) and n.args[2].value==1]
    child_capture=[n for n in calls if n.func.id=='capture_node' and any(k.arg=='time' and isinstance(k.value,ast.Constant) and k.value.value==1 for k in n.keywords)]
    assert len(child_process)==len(child_capture)==1
    assert child_process[0].lineno < child_capture[0].lineno


def test_old_capture_uses_alive_even_when_current_accounting_invalid():
    _,state=fixture();state.update(hatc_cal=torch.zeros(1),lnk_cal=torch.zeros(1),transition_path_id=torch.tensor([7]))
    record=capture_node(state,[dict(parent_candidate_id=7,economic_state_valid=False)],
        split='train',outer=4,time=1,branch=0,model_hashes={})
    assert record['firm_id'].tolist()==[10,11]
    assert record['eligible'].tolist()==[False,False]
    # This proves old eligibility is post-accounting information, not an exit certificate.
    assert record['timing'].startswith('post-entry incumbent-decision')


def test_no_optimization_or_claim_of_readiness():
    r=reproduce();assert r['optimizer_steps']==0 and r['training_performed'] is False
    assert r['default_exit_timing_pass'] is False
