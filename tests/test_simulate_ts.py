"""
Quick smoke test for SimulateTS to inspect generated data shapes/columns.

Runs a tiny simulation on CPU with a randomly initialized PolicyValueModel,
then prints sample rows of firm-level and macro-level outputs.
"""

import sys
from pathlib import Path

import torch

# Allow running as a standalone script
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
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
