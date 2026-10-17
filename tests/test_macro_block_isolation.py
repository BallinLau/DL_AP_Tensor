from __future__ import annotations

import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.run_macro_block_isolation import (
    batch_hash,
    disjoint_path_audit,
    firm_hashes,
    identity_scale_roundtrip,
    install_fc2_guard,
    lag_scan,
    module_hash,
    prepare_batches,
    run_l1_branch,
    sdf_dependency_status,
    set_macro_scope,
    stable_json_hash,
    tensor_hash,
    two_cycle_detect,
    validate_l1_max_outer,
)
from data.tensor_data import TensorTable


class ToyFC1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.FC1_C = torch.nn.Linear(4, 1)
        self.FC1_K = torch.nn.Linear(4, 1)


class ToyMacro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_model = ToyFC1()
        self.sdf_model = torch.nn.Linear(4, 1)
        self.value_model = torch.nn.Linear(3, 1)

    def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
        if x_curr.dim() == 3:
            branches = x_curr.shape[1]
            xp = x_prev.unsqueeze(1).expand(-1, branches, -1)
            cp = hatcf_prev.unsqueeze(1).expand(-1, branches, -1)
            kp = lnkf_prev.unsqueeze(1).expand(-1, branches, -1)
            features = torch.cat([xp, x_curr, cp, kp], dim=-1)
            shape = features.shape[:-1] + (1,)
            c = cp + self.fc1_model.FC1_C(features.reshape(-1, 4)).reshape(shape)
            k = kp + self.fc1_model.FC1_K(features.reshape(-1, 4)).reshape(shape)
            m = torch.ones_like(c)
            return cp[:, 0], c, m, c, k
        features = torch.cat([x_prev, x_curr, hatcf_prev, lnkf_prev], dim=-1)
        c = hatcf_prev + self.fc1_model.FC1_C(features)
        k = lnkf_prev + self.fc1_model.FC1_K(features)
        return c, k, torch.ones_like(c), c, k


def _table(ids):
    rows = torch.tensor([[float(path), 0.0] for path in ids])
    return TensorTable(rows, ["path", "value"])


def _pair_table():
    columns = [
        "path", "t", "branch", "x_t", "x_t1", "Hatcf_t", "LnKF_t",
        "Hatc_t", "LnK_t", "Hatc_t1", "LnK_t1",
    ]
    rows = []
    for path in range(2):
        for time in range(1, 7):
            for branch in range(2):
                rows.append([
                    path, time, branch, 0.1 * time, 0.1 * (time + 1),
                    1.0, 2.0, 1.0 + time, 2.0 + time,
                    1.0 + time + branch * 0.01, 2.0 + time + branch * 0.01,
                ])
    return TensorTable(torch.tensor(rows, dtype=torch.float32), columns)


def test_l0_batch_preparation_has_no_simulation_call():
    assert "run_simulation" not in inspect.getsource(prepare_batches)


def test_frozen_batch_hash_changes_on_mutation():
    batches = [{"parent": torch.zeros(2, 9), "children": [torch.zeros(2, 9)]}]
    before = batch_hash(batches)
    batches[0]["parent"][0, 0] = 1.0
    assert batch_hash(batches) != before


def test_fc1_c_scope_only_enables_c():
    model = ToyMacro()
    assert set_macro_scope(model, "fc1_c") == ["fc1_c"]
    assert all(p.requires_grad for p in model.fc1_model.FC1_C.parameters())
    assert not any(p.requires_grad for p in model.fc1_model.FC1_K.parameters())


def test_fc1_k_scope_only_enables_k():
    model = ToyMacro()
    assert set_macro_scope(model, "fc1_k") == ["fc1_k"]
    assert not any(p.requires_grad for p in model.fc1_model.FC1_C.parameters())
    assert all(p.requires_grad for p in model.fc1_model.FC1_K.parameters())


def test_sdf_scope_freezes_fc1():
    model = ToyMacro()
    assert set_macro_scope(model, "sdf") == ["sdf_model", "value_model"]
    assert not any(p.requires_grad for p in model.fc1_model.parameters())


def test_joint_scope_is_only_macro_model():
    model = ToyMacro()
    assert set(set_macro_scope(model, "joint")) == {"fc1_c", "fc1_k", "sdf_model", "value_model"}
    assert all(p.requires_grad for p in model.parameters())


def test_fc2_guard_raises_and_counts():
    fc2 = torch.nn.Linear(1, 1)
    counter = {"calls": 0}
    install_fc2_guard(fc2, counter)
    with pytest.raises(RuntimeError, match="FC2 MUST NOT BE USED"):
        fc2(torch.ones(1, 1))
    assert counter["calls"] == 1


def test_firm_hash_detects_parameter_change():
    firm = torch.nn.Sequential(torch.nn.Linear(2, 2))
    before = firm_hashes(firm)["all_firm"]
    with torch.no_grad():
        next(firm.parameters()).add_(1.0)
    assert firm_hashes(firm)["all_firm"] != before


def test_l1_source_never_calls_p_root_solver():
    assert "solve_p_block" not in inspect.getsource(run_l1_branch)


def test_l1_source_never_calls_q_root_solver():
    assert "solve_q_block" not in inspect.getsource(run_l1_branch)


def test_l1_source_never_calls_policy_iteration():
    assert "policy_iteration" not in inspect.getsource(run_l1_branch)


def test_crn_fingerprint_is_stable():
    payload = {"seed": 20261017, "transition": 20262028, "entry": 20262029}
    assert stable_json_hash(payload) == stable_json_hash(payload)


def test_probe_bank_hash_is_stable():
    probe = torch.randn(16, 9)
    assert tensor_hash(probe) == tensor_hash(probe.clone())


def test_train_validation_paths_are_disjoint():
    result = disjoint_path_audit(_table([0, 1, 2]), _table([10, 11]))
    assert result["overlap_count"] == 0


def test_physical_scale_identity_roundtrip():
    assert identity_scale_roundtrip(torch.tensor([-2.0, 0.0, 3.0]))


def test_lag_scan_emits_all_requested_offsets(tmp_path: Path):
    output = tmp_path / "scan.csv"
    scan = lag_scan(ToyMacro(), _pair_table(), output)
    assert set(scan["target_offset"]) == {-1, 0, 1, 2, 3}
    assert set(scan["variable"]) == {"Hatc", "LnK"}
    assert output.is_file()


def test_sdf_dependency_degenerate_classification():
    result = sdf_dependency_status("same", "same")
    assert result["experiment_status"] == "DEGENERATE_TO_L0_SDF"
    assert not result["sdf_outer_dependency"]


def test_l1_max_outer_eight_is_strict():
    assert validate_l1_max_outer(8) == 8
    with pytest.raises(ValueError):
        validate_l1_max_outer(9)


def test_two_cycle_detector_synthetic():
    one = torch.ones(16)
    minus_one = -one
    assert two_cycle_detect(one, minus_one, one)


def test_damped_oscillation_is_not_a_two_cycle():
    one = torch.ones(16)
    minus_half = torch.full((16,), -0.5)
    quarter = torch.full((16,), 0.25)
    assert not two_cycle_detect(one, minus_half, quarter)


def test_monotone_damping_is_not_a_two_cycle():
    one = torch.ones(16)
    half = torch.full((16,), 0.5)
    quarter = torch.full((16,), 0.25)
    assert not two_cycle_detect(one, half, quarter)


def test_resume_roundtrip_preserves_frozen_hash(tmp_path: Path):
    model = ToyMacro()
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    before = module_hash(model)
    restored = ToyMacro()
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert module_hash(restored) == before
