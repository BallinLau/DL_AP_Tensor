"""
Quick smoke test for SimulateTS to inspect generated data shapes/columns.

Runs a tiny simulation on CPU with a randomly initialized PolicyValueModel,
then prints sample rows of firm-level and macro-level outputs.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

# Allow running as a standalone script
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from data import simulate_ts_parallel  # noqa: E402
from data import simulate_ts as simulate_ts_module  # noqa: E402
from data.sample import Sample  # noqa: E402
from data.simulate_ts import SimulateTS  # noqa: E402
from models.policy_value import PolicyValueModel  # noqa: E402
from utils.firm_transition import apply_refinancing_policy  # noqa: E402


def build_dummy_models(device: torch.device):
    """Build minimal model dict required by SimulateTS."""
    pv = PolicyValueModel().to(device)
    return {
        'policy_value': pv,
        'sdf_fc1': None,
        'fc2': None,
        'dist_b': None
    }


def _assert_effective_debt_identity(
    b: torch.Tensor,
    eta_next: torch.Tensor,
    bp: torch.Tensor,
    b_next: torch.Tensor,
    atol: float = 1e-6,
) -> None:
    expected = eta_next * bp + (1.0 - eta_next) * b
    torch.testing.assert_close(b_next, expected, atol=atol, rtol=0.0)


def _fake_policy_value_output(firm_state: torch.Tensor):
    n = firm_state.shape[0]
    device = firm_state.device
    zeros = torch.zeros(n, 1, device=device)
    return SimpleNamespace(
        Q=zeros,
        P0=zeros,
        PI=zeros,
        bar_i=zeros,
        bar_z=zeros,
        P=zeros,
        bp0=torch.tensor([[0.2], [0.7]], device=device)[:n],
        bpI=torch.tensor([[0.3], [0.8]], device=device)[:n],
        bp=torch.tensor([[0.25], [0.9]], device=device)[:n],
    )


def test_effective_next_debt_uses_child_eta():
    b = torch.tensor([0.6, 0.4])
    eta = torch.tensor([1.0, 0.0])
    bp = torch.tensor([0.2, 0.9])

    b_next = apply_refinancing_policy(
        b_current=b,
        bp_candidate=bp,
        eta_next=eta,
    )

    torch.testing.assert_close(b_next, torch.tensor([0.2, 0.4]))
    _assert_effective_debt_identity(b, eta, bp, b_next)


def test_parallel_simulation_child_eta_changes_branch_leverage(monkeypatch):
    device = torch.device("cpu")
    eta_draws = [
        torch.zeros(2, 1, device=device),
        torch.ones(2, 1, device=device),
    ]

    def fake_bernoulli(numel, p, device_arg):
        return eta_draws.pop(0).reshape(-1)

    monkeypatch.setattr(simulate_ts_parallel, "sample_bernoulli", fake_bernoulli)
    sim = SimpleNamespace(
        device=device,
        branch_num=2,
        config=Config,
        g=Config.G,
        models={"sdf_fc1": None},
    )
    state = {
        "x": torch.zeros(2, device=device),
        "b": torch.tensor([[0.6], [0.4]], device=device),
        "bp": torch.tensor([[0.2], [0.9]], device=device),
        "z": torch.zeros(2, 1, device=device),
        "eta": torch.tensor([[1.0], [0.0]], device=device),
        "i": torch.zeros(2, 1, device=device),
        "K": torch.ones(2, 1, device=device),
        "hatcf": torch.zeros(2, device=device),
        "lnkf": torch.zeros(2, device=device),
        "M": torch.ones(2, device=device),
        "alive": torch.ones(2, 1, dtype=torch.bool, device=device),
        "entry": torch.zeros(2, 1, device=device),
        "firm_id": torch.arange(2, device=device).reshape(2, 1),
        "next_firm_id": torch.full((2,), 2, device=device),
        "bar_i": torch.zeros(2, 1, device=device),
        "bar_z": torch.zeros(2, 1, device=device),
    }

    branches = simulate_ts_parallel._expand_branches_batched(sim, state)

    torch.testing.assert_close(branches[0]["b"], state["b"])
    torch.testing.assert_close(branches[1]["b"], state["bp"])
    assert not torch.equal(branches[0]["eta"], branches[1]["eta"])


def test_sample_update_child_leverage_uses_each_child_eta():
    sample = Sample(models={}, n_samples=0, n_paths=0)
    df = pd.DataFrame(
        [
            {"path": 0, "ID": "a", "branch": 0, "b": 0.6, "bp": 0.2, "ETA": 0.0},
            {"path": 0, "ID": "a", "branch": 1, "b": 0.0, "bp": 0.0, "ETA": 1.0},
            {"path": 0, "ID": "a", "branch": 2, "b": 0.0, "bp": 0.0, "ETA": 0.0},
            {"path": 0, "ID": "b", "branch": 0, "b": 0.4, "bp": 0.9, "ETA": 1.0},
            {"path": 0, "ID": "b", "branch": 1, "b": 0.0, "bp": 0.0, "ETA": 0.0},
            {"path": 0, "ID": "b", "branch": 2, "b": 0.0, "bp": 0.0, "ETA": 1.0},
        ]
    )

    sample._update_child_leverage(df)

    a_children = df[(df["ID"] == "a") & (df["branch"] > 0)]
    b_children = df[(df["ID"] == "b") & (df["branch"] > 0)]
    assert a_children["b"].round(8).tolist() == [0.2, 0.6]
    assert b_children["b"].round(8).tolist() == [0.4, 0.9]
    assert a_children["ETA"].tolist() == [1.0, 0.0]
    assert b_children["ETA"].tolist() == [0.0, 1.0]


def _make_serial_sim(device: torch.device):
    return SimulateTS(
        models={"policy_value": None, "sdf_fc1": None, "fc2": None},
        config=Config,
        n_paths=1,
        group_size=2,
        horizon=1,
        branch_num=2,
        enable_entry=False,
        enable_exit=False,
        device=device,
    )


def _serial_state(device: torch.device):
    return {
        "x": torch.tensor(0.0, device=device),
        "b": torch.tensor([0.6, 0.4], device=device),
        "bp": torch.tensor([0.2, 0.9], device=device),
        "z": torch.zeros(2, device=device),
        "eta": torch.tensor([1.0, 0.0], device=device),
        "i": torch.zeros(2, device=device),
        "K": torch.ones(2, device=device),
        "hatcf": torch.tensor(0.0, device=device),
        "lnkf": torch.tensor(0.0, device=device),
        "M": torch.tensor(1.0, device=device),
        "hatc_cal": torch.tensor(0.0, device=device),
        "lnk_cal": torch.tensor(0.0, device=device),
        "alive": torch.ones(2, dtype=torch.bool, device=device),
        "entry": torch.zeros(2, device=device),
        "firm_id": torch.arange(2, device=device),
        "ids": ["0", "1"],
        "next_firm_id": torch.tensor(2, device=device),
        "bar_i": torch.zeros(2, device=device),
        "bar_z": torch.zeros(2, device=device),
    }


def test_serial_tensor_expand_branches_uses_child_eta(monkeypatch):
    device = torch.device("cpu")
    eta_draws = [torch.zeros(2, device=device), torch.ones(2, device=device)]

    def fake_bernoulli(numel, p, device_arg):
        return eta_draws.pop(0).reshape(-1)

    monkeypatch.setattr(simulate_ts_module, "sample_bernoulli", fake_bernoulli)
    sim = _make_serial_sim(device)
    branches = sim._expand_branches_tensor(_serial_state(device))

    torch.testing.assert_close(branches[0]["b"], torch.tensor([0.6, 0.4], device=device))
    torch.testing.assert_close(branches[1]["b"], torch.tensor([0.2, 0.9], device=device))
    assert not torch.equal(branches[0]["eta"], branches[1]["eta"])


def test_serial_tensor_node_defers_next_debt_until_child_eta_is_drawn(monkeypatch):
    device = torch.device("cpu")
    sim = _make_serial_sim(device)
    sim.models["policy_value"] = object()

    def fake_forward(model, firm_state):
        return _fake_policy_value_output(firm_state)

    monkeypatch.setattr(
        simulate_ts_module,
        "forward_policy_value_for_simulation",
        fake_forward,
    )

    firm_rows, _ = sim._process_node_tensor(
        _serial_state(device),
        path_idx=0,
        t=0,
        branch_k=-1,
    )
    columns = {name: idx for idx, name in enumerate(sim.FIRM_COLUMNS)}

    assert firm_rows.shape[1] == len(sim.FIRM_COLUMNS)
    for name in ("b_next_p0", "b_next_pi", "b_next_policy"):
        assert torch.isnan(firm_rows[:, columns[name]]).all()


def test_parallel_tensor_node_defers_next_debt_until_child_eta_is_drawn(monkeypatch):
    device = torch.device("cpu")
    sim = _make_serial_sim(device)
    sim.models["policy_value"] = object()
    sim.n_paths = 1

    def fake_forward(model, firm_state):
        return _fake_policy_value_output(firm_state)

    monkeypatch.setattr(
        simulate_ts_parallel,
        "forward_policy_value_for_simulation",
        fake_forward,
    )

    state = {
        "x": torch.zeros(1, device=device),
        "b": torch.tensor([[0.6, 0.4]], device=device),
        "z": torch.zeros(1, 2, device=device),
        "eta": torch.tensor([[1.0, 0.0]], device=device),
        "i": torch.zeros(1, 2, device=device),
        "K": torch.ones(1, 2, device=device),
        "hatcf": torch.zeros(1, device=device),
        "lnkf": torch.zeros(1, device=device),
        "M": torch.ones(1, device=device),
        "alive": torch.ones(1, 2, dtype=torch.bool, device=device),
        "entry": torch.zeros(1, 2, device=device),
        "firm_id": torch.arange(2, device=device).reshape(1, 2),
        "next_firm_id": torch.full((1,), 2, device=device),
        "bar_i": torch.zeros(1, 2, device=device),
        "bar_z": torch.zeros(1, 2, device=device),
        "bp": torch.tensor([[0.6, 0.4]], device=device),
    }

    firm_rows, _ = simulate_ts_parallel._process_node_batched(
        sim,
        state,
        t=0,
        branch_k=-1,
    )
    columns = {name: idx for idx, name in enumerate(sim.FIRM_COLUMNS)}

    assert firm_rows.shape[1] == len(sim.FIRM_COLUMNS)
    for name in ("b_next_p0", "b_next_pi", "b_next_policy"):
        assert torch.isnan(firm_rows[:, columns[name]]).all()


def test_serial_legacy_expand_branches_uses_child_eta(monkeypatch):
    device = torch.device("cpu")
    eta_draws = [torch.zeros(2, device=device), torch.ones(2, device=device)]

    def fake_bernoulli(numel, p, device_arg):
        return eta_draws.pop(0).reshape(-1)

    monkeypatch.setattr(simulate_ts_module, "sample_bernoulli", fake_bernoulli)
    sim = _make_serial_sim(device)
    branches = sim._expand_branches(_serial_state(device), t=0)

    torch.testing.assert_close(branches[0]["b"], torch.tensor([0.6, 0.4], device=device))
    torch.testing.assert_close(branches[1]["b"], torch.tensor([0.2, 0.9], device=device))
    assert not torch.equal(branches[0]["eta"], branches[1]["eta"])


def test_simulation_dataframe_reports_effective_next_debt():
    device = torch.device("cpu")
    Config.DEVICE = device

    sim = SimulateTS(
        models=build_dummy_models(device),
        config=Config,
        n_paths=1,
        group_size=3,
        horizon=1,
        branch_num=2,
        enable_entry=False,
        enable_exit=False,
        device=device,
    )

    df_firm, _ = sim.simulate()

    required = {"b_next_p0", "b_next_pi", "b_next_policy"}
    assert required.issubset(set(df_firm.columns))
    parents = df_firm[df_firm["branch"] == -1]
    nonparents = df_firm[df_firm["branch"] >= 0]
    assert parents["b_next_policy"].notna().all()
    assert nonparents["b_next_policy"].isna().all()

    children = df_firm[df_firm["branch"] == 0]
    merged = parents.merge(
        children,
        left_on=["path", "ID"],
        right_on=["path", "ID"],
        suffixes=("_parent", "_child"),
    )
    merged = merged[merged["t_child"] == merged["t_parent"] + 1]
    assert not merged.empty
    expected = (
        merged["ETA_child"] * merged["bp_parent"]
        + (1.0 - merged["ETA_child"]) * merged["b_parent"]
    )
    assert (merged["b_next_policy_parent"] - expected).abs().max() < 1e-6
    max_transition_error = (
        merged["b_next_policy_parent"] - merged["b_child"]
    ).abs().max()
    assert max_transition_error < 1e-6


def main():
    # Force CPU for reproducibility/debugging
    device = torch.device('cpu')
    Config.DEVICE = device

    models = build_dummy_models(device)

    sim = SimulateTS(
        models=models,
        config=Config,
        n_paths=1,
        group_size=3,
        horizon=2,
        branch_num=2,
        enable_entry=False,
        enable_exit=False,
        device=device
    )

    df_firm, df_macro = sim.simulate()

    print("\n--- Firm-level (first 10 rows) ---")
    print(df_firm.head(10))
    print("\nColumns:", df_firm.columns.tolist())

    print("\n--- Macro-level (all rows) ---")
    print(df_macro)
    print("\nMacro Columns:", df_macro.columns.tolist())


if __name__ == "__main__":
    main()
