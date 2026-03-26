"""
Compare DL_AP_Local vs DL_AP_Tensor data generation consistency.

This script runs Sample/SimulateTS in both repos with the same seed, saves raw
CSV outputs, summarizes key moments, and plots overlay histograms for a broad
set of variables.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKER_CODE = r"""
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import sys

repo = sys.argv[1]
out_dir = Path(sys.argv[2])
seed = int(sys.argv[3])
n_paths_sdf = int(sys.argv[4])
n_paths_pv = int(sys.argv[5])
n_paths_sim = int(sys.argv[6])
group_size_sim = int(sys.argv[7])
horizon_sim = int(sys.argv[8])
entry_flag = bool(int(sys.argv[9]))
exit_flag = bool(int(sys.argv[10]))

sys.path.insert(0, repo)
from data.sample import Sample
from data.simulate_ts import SimulateTS
from config import Config

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cpu")


class DummyPV(torch.nn.Module):
    def forward(self, x):
        b = x[:, 0:1]
        z = x[:, 1:2]
        q = (0.30 + 0.10 * torch.tanh(z) - 0.08 * b).clamp(min=0.0, max=1.0)
        p0 = (0.15 + 0.08 * torch.tanh(z) - 0.06 * b).clamp(min=0.0)
        pi = (0.18 + 0.09 * torch.tanh(z) - 0.05 * b).clamp(min=0.0)
        bar_i_cond = torch.sigmoid(0.10 + 0.20 * z)
        chi = 1.0 - torch.sigmoid(-1.00 + 1.20 * b - 0.30 * z)
        bar_i = chi * bar_i_cond
        bar_z = torch.sigmoid(-1.00 + 1.20 * b - 0.30 * z)
        p = ((1 - bar_i) * p0 + bar_i * pi) * (1 - bar_z)
        bp0 = torch.sigmoid(-0.10 + 0.90 * b)
        bpI = torch.sigmoid(0.00 + 0.80 * b)
        bp = ((1 - bar_i) * bp0 + bar_i * bpI).clamp(0.0, 1.0)
        return SimpleNamespace(
            Q=q,
            bp0=bp0,
            bpI=bpI,
            V0=p0,
            VI=pi,
            Vhat=p,
            chi=chi,
            bar_i_cond=bar_i_cond,
            P0=p0,
            PI=pi,
            bar_i=bar_i,
            bar_z=bar_z,
            P=p,
            Phat=p,
            bp=bp,
        )


def summarize(df: pd.DataFrame, cols):
    out = {"rows": int(len(df)), "cols": int(df.shape[1])}
    stats = {}
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            stats[c] = {
                "mean": float(np.nanmean(s)),
                "std": float(np.nanstd(s)),
                "p50": float(np.nanpercentile(s, 50)),
            }
    out["stats"] = stats
    if "branch" in df.columns:
        vc = df["branch"].value_counts(dropna=False).sort_index()
        out["branch_counts"] = {str(k): int(v) for k, v in vc.items()}
    return out


sample_sdf = Sample(
    models={},
    config=Config,
    n_samples=None,
    n_paths=n_paths_sdf,
    group_size=2,
    branch_num=2,
    data_mode="sample",
    device=device,
)
df_sample_sdf = sample_sdf.build_sdf_fc1_df()

sample_pv = Sample(
    models={},
    config=Config,
    n_samples=None,
    n_paths=n_paths_pv,
    group_size=2,
    branch_num=2,
    data_mode="sample",
    device=device,
)
df_sample_pv = sample_pv.build_policy_value_df()

models = {"policy_value": DummyPV(), "sdf_fc1": None}
sim = SimulateTS(
    models=models,
    config=Config,
    n_paths=n_paths_sim,
    group_size=group_size_sim,
    horizon=horizon_sim,
    branch_num=2,
    enable_entry=entry_flag,
    enable_exit=exit_flag,
    device=device,
)
df_sim_firm, df_sim_macro = sim.simulate()

out_dir.mkdir(parents=True, exist_ok=True)
df_sample_sdf.to_csv(out_dir / "sample_sdf.csv", index=False)
df_sample_pv.to_csv(out_dir / "sample_pv.csv", index=False)
df_sim_firm.to_csv(out_dir / "sim_firm.csv", index=False)
df_sim_macro.to_csv(out_dir / "sim_macro.csv", index=False)

summary = {
    "sample_sdf": summarize(df_sample_sdf, ["x_t", "x_t1", "Hatcf_t", "LnKF_t"]),
    "sample_pv": summarize(df_sample_pv, ["b", "z", "ETA", "i", "x", "Hatcf", "LnKF", "K", "M"]),
    "sim_firm": summarize(
        df_sim_firm,
        ["b", "z", "ETA", "i", "x", "Hatcf", "LnKF", "K", "M", "Q", "P0", "PI", "Bar_i", "Bar_z", "P", "bp", "Y", "I", "Phi", "C"],
    ),
    "sim_macro": summarize(df_sim_macro, ["K", "C", "LnK", "Hatc", "n_firms", "M", "x", "hatcf", "lnkf"]),
}
print(json.dumps(summary, ensure_ascii=False))
"""


PLOT_VARS: Dict[str, List[str]] = {
    "sample_sdf": ["x_t", "x_t1", "Hatcf_t", "LnKF_t"],
    "sample_pv": ["b", "z", "ETA", "i", "x", "Hatcf", "LnKF", "K", "M"],
    "sim_firm": [
        "b", "z", "ETA", "i", "x", "Hatcf", "LnKF", "K", "M",
        "Q", "P0", "PI", "Bar_i", "Bar_z", "P", "bp", "Y", "I", "Phi", "C",
    ],
    "sim_macro": ["K", "C", "LnK", "Hatc", "n_firms", "M", "x", "hatcf", "lnkf"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Local vs Tensor data consistency and plot distributions.")
    p.add_argument("--local-repo", type=Path, default=Path("/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local"))
    p.add_argument("--tensor-repo", type=Path, default=Path("/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor"))
    p.add_argument("--python-bin", type=str, default="/Users/ballinliu/anaconda3/bin/python")
    p.add_argument("--out-dir", type=Path, default=Path("/Users/ballinliu/Desktop/PHD/Project1/cachedir/consistency_plots"))
    p.add_argument("--seed", type=int, default=20260318)
    p.add_argument("--n-paths-sdf", type=int, default=1024)
    p.add_argument("--n-paths-pv", type=int, default=512)
    p.add_argument("--n-paths-sim", type=int, default=100)
    p.add_argument("--group-size-sim", type=int, default=30)
    p.add_argument("--horizon-sim", type=int, default=4)
    p.add_argument("--enable-entry", action="store_true", default=False)
    p.add_argument("--enable-exit", action="store_true", default=False)
    p.add_argument("--bins", type=int, default=60)
    p.add_argument("--max-points", type=int, default=120000)
    return p.parse_args()


def run_one_repo(
    python_bin: str,
    repo: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Dict:
    cmd = [
        python_bin,
        "-c",
        WORKER_CODE,
        str(repo),
        str(out_dir),
        str(args.seed),
        str(args.n_paths_sdf),
        str(args.n_paths_pv),
        str(args.n_paths_sim),
        str(args.group_size_sim),
        str(args.horizon_sim),
        "1" if args.enable_entry else "0",
        "1" if args.enable_exit else "0",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _sample_series(s: pd.Series, max_points: int, seed: int) -> np.ndarray:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) > max_points:
        s = s.sample(n=max_points, random_state=seed)
    return s.to_numpy(dtype=float)


def plot_overlay_hist(
    df_local: pd.DataFrame,
    df_tensor: pd.DataFrame,
    vars_list: List[str],
    title: str,
    out_path: Path,
    bins: int,
    max_points: int,
    seed: int,
) -> None:
    valid_vars = [v for v in vars_list if v in df_local.columns and v in df_tensor.columns]
    if not valid_vars:
        return

    n = len(valid_vars)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, var in enumerate(valid_vars):
        ax = axes_flat[i]
        a = _sample_series(df_local[var], max_points=max_points, seed=seed)
        b = _sample_series(df_tensor[var], max_points=max_points, seed=seed + 1)
        if a.size == 0 or b.size == 0:
            ax.set_visible(False)
            continue

        lo = float(np.nanmin([np.nanmin(a), np.nanmin(b)]))
        hi = float(np.nanmax([np.nanmax(a), np.nanmax(b)]))
        if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
            lo, hi = lo - 1.0, hi + 1.0

        ax.hist(a, bins=bins, alpha=0.45, density=True, label="DL_AP_Local", range=(lo, hi))
        ax.hist(b, bins=bins, alpha=0.45, density=True, label="DL_AP_Tensor", range=(lo, hi))
        ax.set_title(var)
        ax.grid(alpha=0.25)

    for j in range(len(valid_vars), len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    local_dir = out_dir / "local"
    tensor_dir = out_dir / "tensor"
    plots_dir = out_dir / "plots"
    local_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_local = run_one_repo(args.python_bin, args.local_repo, local_dir, args)
    summary_tensor = run_one_repo(args.python_bin, args.tensor_repo, tensor_dir, args)

    for ds_name, var_list in PLOT_VARS.items():
        df_local = pd.read_csv(local_dir / f"{ds_name}.csv")
        df_tensor = pd.read_csv(tensor_dir / f"{ds_name}.csv")
        plot_overlay_hist(
            df_local=df_local,
            df_tensor=df_tensor,
            vars_list=var_list,
            title=f"{ds_name} distributions: Local vs Tensor",
            out_path=plots_dir / f"{ds_name}_dist_overlay.png",
            bins=args.bins,
            max_points=args.max_points,
            seed=args.seed,
        )

    payload = {
        "config": {
            "seed": args.seed,
            "n_paths_sdf": args.n_paths_sdf,
            "n_paths_pv": args.n_paths_pv,
            "n_paths_sim": args.n_paths_sim,
            "group_size_sim": args.group_size_sim,
            "horizon_sim": args.horizon_sim,
            "enable_entry": args.enable_entry,
            "enable_exit": args.enable_exit,
        },
        "local_summary": summary_local,
        "tensor_summary": summary_tensor,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] Outputs saved to: {out_dir}")
    print(f"[OK] Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
