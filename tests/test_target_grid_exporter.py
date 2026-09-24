import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models import PolicyValueModel
from analysis.economic_config import AnalysisEconomicConfig


def _write_policy_checkpoint(run_root: Path, *, zeta: float = 0.20) -> None:
    checkpoint_dir = run_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    model = PolicyValueModel()
    torch.save(model.state_dict(), checkpoint_dir / "ep0_policy_value.pt")
    analysis_dir = run_root / "checkpoints_analysis"
    analysis_dir.mkdir(parents=True)
    config_snapshot = AnalysisEconomicConfig.from_current_config().to_dict()
    config_snapshot["ZETA"] = zeta
    torch.save(
        {"config_snapshot": config_snapshot},
        analysis_dir / "ep0_combined.pt",
    )


def _write_firm_stage_pickle(path: Path) -> None:
    rows = []
    for idx in range(3):
        common = {
            "path": idx,
            "ID": f"firm-{idx}",
            "b": 0.15 + 0.05 * idx,
            "z": -0.1 + 0.1 * idx,
            "ETA": 1.0,
            "i": 0.05,
            "x": 0.1,
            "Hatcf": 0.02,
            "LnKF": 0.1,
            "M": 0.98,
        }
        rows.append({**common, "t": 0, "branch": -1})
        for branch in range(2):
            rows.append(
                {
                    **common,
                    "t": 1,
                    "branch": branch,
                    "b": common["b"] + 0.01 * branch,
                    "z": common["z"] + 0.02 * branch,
                    "M": 0.97 + 0.01 * branch,
                }
            )
    pd.DataFrame(rows).to_pickle(path)


def test_exporter_branch_loop_wires_mix_survival_weight(tmp_path):
    run_root = tmp_path / "run"
    output_dir = tmp_path / "out"
    firm_pkl = tmp_path / "firm.pkl"
    _write_policy_checkpoint(run_root)
    _write_firm_stage_pickle(firm_pkl)

    cmd = [
        sys.executable,
        "experiments/export_target_grid_decomposition.py",
        "--run-root",
        str(run_root),
        "--episode",
        "0",
        # fixture 写的是裸 state_dict（run root 无 metadata/），且模型为默认 direct-Q。
        "--raw-q-parameterization",
        "direct",
        "--firm-pkl",
        str(firm_pkl),
        "--n-states",
        "2",
        "--device",
        "cpu",
        "--batch-size",
        "8",
        "--n-branches",
        "2",
        "--output-dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Selected active states: 2" in completed.stdout

    long_df = pd.read_csv(output_dir / "ep0_target_grid_decomposition_long.csv")
    summary_df = pd.read_csv(output_dir / "ep0_target_grid_decomposition_summary.csv")

    assert set(summary_df["branch"]) == {"p0", "pi", "mix"}
    assert set(long_df["branch"]) == {"p0", "pi", "mix"}
    assert "mix_survival_weight" in summary_df.columns
    assert "mix_survival_weight" not in long_df.columns
    assert summary_df["mix_survival_weight"].between(0.0, 1.0).all()
    assert set(summary_df["eta_integration_mode"]) == {"exact"}
    assert set(summary_df["eta_probability"]) == {0.20}
    assert (summary_df["eta_next_active_share"] - 0.20).abs().max() < 1e-7
    assert set(summary_df["economic_config_source"]) == {"checkpoint"}
