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
from data.sample import Sample  # noqa: E402
from data.simulate_ts import SimulateTS  # noqa: E402
from models.policy_value import PolicyValueModel  # noqa: E402


def build_dummy_models(device: torch.device):
    """Build minimal model dict required by SimulateTS."""
    pv = PolicyValueModel().to(device)
    return {
        'policy_value': pv,
        'sdf_fc1': None,
        'fc2': None,
        'dist_b': None
    }


def test_parallel_simulation_next_eta_does_not_change_branch_leverage(monkeypatch):
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

    expected_b = torch.tensor([[0.2], [0.4]], device=device)
    torch.testing.assert_close(branches[0]["b"], expected_b)
    torch.testing.assert_close(branches[1]["b"], expected_b)
    assert not torch.equal(branches[0]["eta"], branches[1]["eta"])


def test_sample_update_child_leverage_uses_parent_eta_not_child_eta():
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
    assert set(a_children["b"].round(8).tolist()) == {0.6}
    assert set(b_children["b"].round(8).tolist()) == {0.9}
    assert a_children["ETA"].tolist() == [1.0, 0.0]
    assert b_children["ETA"].tolist() == [0.0, 1.0]


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
