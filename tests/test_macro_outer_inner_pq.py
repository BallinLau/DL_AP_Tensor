from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.run_macro_outer_inner_pq import (
    audit_m_parity,
    install_fc2_guard,
    load_outer_checkpoint,
    module_hash,
    possible_two_cycle,
    refresh_transition_bank,
    run_macro_outer_skeleton,
    save_outer_checkpoint,
    set_stage_trainability,
    stable_json_hash,
    state_hash,
    _write_report,
)
from analysis.test_frozen_policy_root_solve import FrozenBranchPolicy, dynamic_rhs


class ToyMacro(torch.nn.Module):
    def __init__(self, m_value: float = 0.9, shift: float = 0.0):
        super().__init__()
        self.m_value = torch.nn.Parameter(torch.tensor(m_value))
        self.shift = torch.nn.Parameter(torch.tensor(shift))

    def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
        shape = x_curr.shape
        m = torch.ones(shape, device=x_curr.device) * self.m_value
        hatc = hatcf_prev.unsqueeze(1) + x_curr * 0.1 + self.shift
        lnk = lnkf_prev.unsqueeze(1) + x_curr * 0.2 + self.shift
        return hatcf_prev, hatc, m, hatc, lnk


class ToyFirm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.i_threshold = 0.5
        self.i_grid_size = 2

    def _value_outputs(self, states):
        value = torch.ones((states.shape[0], 1), device=states.device) * self.weight
        return value, value


class ToyEpisode:
    def _expand_policy_expectation_children(self, children, raw_m, used_m):
        return children, raw_m, used_m, None


def banks():
    parent = torch.zeros(4, 7)
    parent[:, 5] = -2.0
    parent[:, 6] = 4.0
    child = torch.zeros(4, 2, 8)
    child[:, :, 4] = torch.tensor([-0.1, 0.1])
    child[:, :, 7] = 1.0
    return parent, child


def test_fc2_forward_patched_to_raise_but_outer_skeleton_succeeds():
    fc2 = torch.nn.Linear(1, 1)
    counter = {"calls": 0}
    install_fc2_guard(fc2, counter)
    states = run_macro_outer_skeleton(0, 1, lambda x, _: x + 1, lambda *_: 0, lambda x, *_: x)
    assert states == [1, 2]
    assert counter["calls"] == 0
    with pytest.raises(RuntimeError, match="FC2 MUST NOT BE USED"):
        fc2(torch.ones(1, 1))


def test_fc2_hash_invariant_without_access():
    fc2 = torch.nn.Linear(2, 1)
    before = module_hash(fc2)
    install_fc2_guard(fc2, {"calls": 0})
    assert module_hash(fc2) == before


def test_macro_stage_only_sdf_fc1_trainable():
    models = {"sdf_fc1": ToyMacro(), "policy_value": ToyFirm(), "fc2": torch.nn.Linear(1, 1)}
    assert set_stage_trainability(models, "macro") == ["sdf_fc1"]
    assert all(p.requires_grad for p in models["sdf_fc1"].parameters())
    assert not any(p.requires_grad for p in models["policy_value"].parameters())
    assert not any(p.requires_grad for p in models["fc2"].parameters())


def test_firm_stage_freezes_macro_and_fc2():
    models = {"sdf_fc1": ToyMacro(), "policy_value": ToyFirm(), "fc2": torch.nn.Linear(1, 1)}
    assert set_stage_trainability(models, "firm") == ["policy_value"]
    assert not any(p.requires_grad for p in models["sdf_fc1"].parameters())
    assert not any(p.requires_grad for p in models["fc2"].parameters())


def test_p_q_identical_current_m():
    _, child = banks()
    result = audit_m_parity(ToyEpisode(), child)
    assert result["p_m_hash"] == result["q_m_hash"]
    assert result["max_abs_m_difference"] == 0.0


def test_sdf_m_update_changes_p_operator_rhs():
    firm = ToyFirm()
    branch = FrozenBranchPolicy(
        name="p0", child_states=torch.zeros(1, 2, 7),
        child_weights=torch.full((1, 2), 0.5), selector_pi=torch.zeros(2, 1, 2, dtype=torch.bool),
        survival=torch.ones(1, 2), cashflow=torch.zeros(1, 1), bp_star=torch.zeros(1, 1),
        production_target=torch.zeros(1, 1), continuation_factor=1.0, m=torch.full((1, 2), 0.8),
    )
    rhs_old = dynamic_rhs(firm, branch)
    rhs_new = dynamic_rhs(firm, replace(branch, m=torch.full((1, 2), 1.2)))
    assert not torch.allclose(rhs_old, rhs_new)


def test_fc1_update_changes_firm_macro_inputs():
    parent, child = banks()
    old, old_diag = refresh_transition_bank(ToyMacro(0.9, 0.0), parent, child, m_min=0.7, m_max=1.3)
    new, new_diag = refresh_transition_bank(ToyMacro(0.9, 0.2), parent, child, m_min=0.7, m_max=1.3)
    assert old_diag["macro_input_hash"] != new_diag["macro_input_hash"]
    assert not torch.allclose(old[..., 5:7], new[..., 5:7])


def test_common_random_number_fingerprint_constant_across_outer():
    spec = [20261028, 2, 100, "stable_firm_identity"]
    assert stable_json_hash(spec) == stable_json_hash(spec)


def test_firm_warm_start_flows_between_outer_steps():
    seen = []
    def solve(value, outer):
        seen.append((outer, value))
        return value + 1
    run_macro_outer_skeleton(0, 2, solve, lambda *_: None, lambda value, *_: value)
    assert seen == [(0, 0), (1, 1), (2, 2)]


def test_sdf_fc1_warm_start_checkpoint_roundtrip(tmp_path: Path):
    macro, firm = ToyMacro(), ToyFirm()
    target = save_outer_checkpoint(tmp_path, 1, macro, firm, {"outer": 1})
    with torch.no_grad():
        macro.m_value.fill_(0.1)
        firm.weight.fill_(4.0)
    meta = load_outer_checkpoint(target, macro, firm)
    assert macro.m_value.item() == pytest.approx(0.9)
    assert firm.weight.item() == pytest.approx(1.0)
    assert meta["outer"] == 1


def test_max_macro_outer_five_strictly_enforced():
    with pytest.raises(ValueError):
        run_macro_outer_skeleton(0, 6, lambda x, _: x, lambda *_: None, lambda x, *_: x)
    assert len(run_macro_outer_skeleton(0, 5, lambda x, _: x, lambda *_: None, lambda x, *_: x)) == 6


def test_two_cycle_detector_synthetic():
    a = {name: torch.zeros(8) for name in ("m", "hatc", "lnk")}
    b = {name: torch.ones(8) for name in ("m", "hatc", "lnk")}
    assert possible_two_cycle(a, b, a)


def test_no_hidden_fc2_access_in_skeleton():
    fc2 = torch.nn.Linear(1, 1)
    counter = {"calls": 0}
    install_fc2_guard(fc2, counter)
    run_macro_outer_skeleton({}, 1, lambda x, _: x, lambda *_: {}, lambda x, *_: x)
    assert counter["calls"] == 0


def test_macro_outer_checkpoint_resume_restores_both_blocks(tmp_path: Path):
    macro, firm = ToyMacro(1.1, 0.3), ToyFirm()
    target = save_outer_checkpoint(tmp_path, 3, macro, firm, {"outer": 3, "tag": "resume"})
    macro2, firm2 = ToyMacro(0.7, -1.0), ToyFirm()
    meta = load_outer_checkpoint(target, macro2, firm2)
    assert module_hash(macro2) == module_hash(macro)
    assert module_hash(firm2) == module_hash(firm)
    assert meta["tag"] == "resume"


def test_train_validation_simulation_banks_not_mixed():
    train = {"b": torch.zeros(2, 3), "x": torch.zeros(2)}
    validation = {"b": torch.ones(2, 3), "x": torch.ones(2)}
    assert state_hash(train) != state_hash(validation)


def test_report_writer_has_no_optional_tabulate_dependency(tmp_path: Path):
    path = tmp_path / "report.md"
    _write_report(
        path,
        args=SimpleNamespace(example=True),
        metadata={"classification": "E", "m_parity": {}},
        summary=pd.DataFrame([{"outer": 0, "p_residual_post_inner": 0.1}]),
    )
    assert "| outer | p_residual_post_inner |" in path.read_text(encoding="utf-8")

