from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from data.simulate_ts import SimulateTS  # noqa: E402
from models.policy_value import PolicyValueOutput  # noqa: E402
from training.bp_grid_teacher import BPGridTeacher  # noqa: E402


class SimulationOnlyPolicyValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.simulation_calls = 0

    def forward(self, firm_state):
        raise AssertionError("simulation must call forward_simulation(), not generic forward()")

    def forward_simulation(self, firm_state):
        self.simulation_calls += 1
        n = firm_state.shape[0]
        device = firm_state.device
        dtype = firm_state.dtype
        zeros = torch.zeros(n, 1, device=device, dtype=dtype)
        ones = torch.ones(n, 1, device=device, dtype=dtype)
        bp0 = torch.full((n, 1), 0.2, device=device, dtype=dtype)
        bpI = torch.full((n, 1), 0.4, device=device, dtype=dtype)
        bar_i_cond = torch.full((n, 1), 0.5, device=device, dtype=dtype)
        survival_prob = ones
        bp_cond = bar_i_cond * bpI + (1.0 - bar_i_cond) * bp0
        return PolicyValueOutput(
            Q=zeros,
            bp0=bp0,
            bpI=bpI,
            P0=ones,
            PI=ones,
            bar_i=bar_i_cond,
            bar_z=zeros,
            P=ones,
            Phat=ones,
            bp=bp_cond,
            V0=ones,
            VI=ones,
            bar_i_cond=bar_i_cond,
            bar_i_eff=bar_i_cond,
            survival_prob=survival_prob,
            bp_cond=bp_cond,
        )


class RecordingSdfFc1(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
        self.calls.append(
            {
                "x_prev": x_prev.detach().clone(),
                "x_curr": x_curr.detach().clone(),
                "hatcf_prev": hatcf_prev.detach().clone(),
                "lnkf_prev": lnkf_prev.detach().clone(),
            }
        )
        m = torch.ones_like(hatcf_prev, dtype=torch.float32)
        hatcf_next = hatcf_prev + 0.01
        lnkf_next = lnkf_prev + 0.02
        return m, m, m, hatcf_next, lnkf_next


def test_simulation_uses_forward_simulation_and_not_grid_teacher():
    device = torch.device("cpu")
    Config.DEVICE = device
    pv_model = SimulationOnlyPolicyValue().to(device)
    models = {
        "policy_value": pv_model,
        "sdf_fc1": None,
        "fc2": None,
        "dist_b": None,
    }
    sim = SimulateTS(
        models=models,
        config=Config,
        n_paths=1,
        group_size=2,
        horizon=1,
        branch_num=1,
        enable_entry=False,
        enable_exit=False,
        device=device,
    )

    original_compute = BPGridTeacher.compute

    def fail_if_called(*args, **kwargs):
        raise AssertionError("simulation must not call BPGridTeacher")

    BPGridTeacher.compute = fail_if_called
    try:
        out = sim.simulate_tensor()
    finally:
        BPGridTeacher.compute = original_compute

    assert pv_model.simulation_calls > 0
    bp_idx = out.firm.columns.index("bp")
    assert torch.allclose(out.firm.data[:, bp_idx], torch.full_like(out.firm.data[:, bp_idx], 0.3))


def test_fc1_child_forecast_uses_calculated_parent_macro_state():
    device = torch.device("cpu")
    Config.DEVICE = device
    pv_model = SimulationOnlyPolicyValue().to(device)
    sdf_fc1 = RecordingSdfFc1()
    models = {
        "policy_value": pv_model,
        "sdf_fc1": sdf_fc1,
        "fc2": None,
        "dist_b": None,
    }
    sim = SimulateTS(
        models=models,
        config=Config,
        n_paths=1,
        group_size=2,
        horizon=1,
        branch_num=1,
        enable_entry=False,
        enable_exit=False,
        device=device,
    )

    out = sim.simulate_tensor()

    assert sdf_fc1.calls, "simulation should call SDF/FC1 during branch expansion"
    call = sdf_fc1.calls[0]
    parent_macro = out.macro.data[0]
    lnk_idx = out.macro.columns.index("LnK")
    hatc_idx = out.macro.columns.index("Hatc")
    forecast_hat_idx = out.macro.columns.index("hatcf")
    forecast_lnk_idx = out.macro.columns.index("lnkf")

    assert torch.allclose(call["hatcf_prev"].reshape(-1), parent_macro[hatc_idx].reshape(-1))
    assert torch.allclose(call["lnkf_prev"].reshape(-1), parent_macro[lnk_idx].reshape(-1))
    assert not torch.allclose(call["hatcf_prev"].reshape(-1), parent_macro[forecast_hat_idx].reshape(-1))


if __name__ == "__main__":
    test_simulation_uses_forward_simulation_and_not_grid_teacher()
    test_fc1_child_forecast_uses_calculated_parent_macro_state()
