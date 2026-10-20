"""CPU forward-only blocker reproduction. NOT a dynamic sampling implementation.

Uses the unchanged production branch transition, no model or optimizer.
The synthetic parent is explicitly post-decision, before the omitted exit.
"""
from copy import deepcopy
from types import SimpleNamespace
import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
from data.simulate_ts_parallel import _expand_branches_batched, _apply_exit_batched


def fixture():
    device = torch.device('cpu')
    sim = SimpleNamespace(device=device, branch_num=2, config=Config, g=Config.G,
        models={'sdf_fc1': None}, aggregation_mode='legacy_mixed_replay',
        transition_rng_mode='stable_firm_identity', common_transition_seed=202610214,
        enable_exit=True)
    state = dict(x=torch.zeros(1), b=torch.tensor([[.6, .4, .8]]),
        bp=torch.tensor([[.2, .9, .3]]), z=torch.zeros(1,3),
        eta=torch.tensor([[1.,0.,1.]]), i=torch.zeros(1,3), K=torch.tensor([[1.,2.,3.]]),
        hatcf=torch.zeros(1), lnkf=torch.zeros(1), M=torch.ones(1),
        alive=torch.tensor([[True,True,False]]), entry=torch.zeros(1,3),
        firm_id=torch.tensor([[10,11,12]]), next_firm_id=torch.tensor([13]),
        bar_i=torch.zeros(1,3), bar_z=torch.tensor([[1.,0.,0.]]))
    return sim, state


@torch.no_grad()
def reproduce():
    sim, parent = fixture()
    children = _expand_branches_batched(sim, deepcopy(parent), child_t=1)
    defaulted = parent['alive'] & (parent['bar_z'] >= .5)
    rows = []
    for branch, child in enumerate(children):
        leaked = defaulted & child['alive']
        # Diagnostic counterfactual on a COPY only: demonstrates the existing
        # exit function's effect, never inserted into production execution.
        after_exit = _apply_exit_batched(deepcopy(child))
        rows.append(dict(branch=branch, parent_defaulted_ids=parent['firm_id'][defaulted].tolist(),
            child_alive_ids=child['firm_id'][child['alive']].tolist(),
            previous_default_still_alive_ids=child['firm_id'][leaked].tolist(),
            child_b=child['b'].tolist(), child_eta=child['eta'].tolist(),
            exit_on_copy_alive_ids=after_exit['firm_id'][after_exit['alive']].tolist()))
    return dict(status='BLOCKED_PREVIOUS_DEFAULT_NOT_REMOVED_BEFORE_CHILD_DECISION',
        evidence_kind='SYNTHETIC_CPU_PRODUCTION_TRANSITION_FORWARD_ONLY',
        source_node='post-parent-decision before exit', parent_eta=parent['eta'].tolist(),
        parent_b=parent['b'].tolist(), parent_bp=parent['bp'].tolist(), branches=rows,
        realized_b_eta_t_parity=all(torch.equal(c['b'],torch.tensor([[.2,.4,.3]])) for c in children),
        default_exit_timing_pass=False, production_code_changed=False, optimizer_steps=0,
        economic_equations_changed=False, training_performed=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(reproduce(),indent=2)+'\n')
    print(args.output)
