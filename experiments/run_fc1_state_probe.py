"""
Standalone FC1 state-sufficiency probe.

This script compares two small feedforward probes on simulated macro transition
data to identify whether cross-sectional summaries add predictive information:

1. Baseline:
   [x_t, x_{t+1}, Hatc_t, LnK_t] -> [Hatc_{t+1}, LnK_{t+1}]
2. Augmented:
   [x_t, x_{t+1}, Hatc_t, LnK_t, b_mean_t, b_std_t, z_mean_t, z_std_t]
   -> [Hatc_{t+1}, LnK_{t+1}]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.data_utils import build_sdf_pairs_from_macro_ts  # noqa: E402
from models.base import MLP  # noqa: E402


def _fit_stats_np(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {
            "n": 0.0,
            "r2": float("nan"),
            "corr": float("nan"),
            "slope": float("nan"),
            "intercept": float("nan"),
            "std_ratio": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
        }
    yt = y_true[mask].astype(np.float64)
    yp = y_pred[mask].astype(np.float64)
    ss_res = float(np.sum((yt - yp) ** 2))
    yt_mean = float(np.mean(yt))
    ss_tot = float(np.sum((yt - yt_mean) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    corr = float(np.corrcoef(yt, yp)[0, 1]) if yt.shape[0] >= 2 else float("nan")
    x_mean = float(np.mean(yt))
    y_mean = float(np.mean(yp))
    x_var = float(np.sum((yt - x_mean) ** 2))
    if x_var <= 0:
        slope = float("nan")
        intercept = float("nan")
    else:
        slope = float(np.sum((yt - x_mean) * (yp - y_mean)) / x_var)
        intercept = float(y_mean - slope * x_mean)
    std_true = float(np.std(yt))
    std_pred = float(np.std(yp))
    std_ratio = float("nan") if std_true <= 0 else std_pred / std_true
    rmse = float(math.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    return {
        "n": float(yt.shape[0]),
        "r2": r2,
        "corr": corr,
        "slope": slope,
        "intercept": intercept,
        "std_ratio": std_ratio,
        "rmse": rmse,
        "mae": mae,
    }


def _prefixed_stats(prefix: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in _fit_stats_np(y_true, y_pred).items()}


def _ensure_numeric_t(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["t"] = pd.to_numeric(out["t"], errors="coerce")
    return out


def _parent_branch_value(branch_values: pd.Series) -> int:
    vals = pd.to_numeric(branch_values, errors="coerce").dropna().astype(int)
    if vals.empty:
        raise ValueError("No valid branch values found.")
    return -1 if (vals < 0).any() else 0


def build_parent_summaries(df_firm: pd.DataFrame) -> pd.DataFrame:
    required = {"path", "t", "branch", "b", "z"}
    missing = required - set(df_firm.columns)
    if missing:
        raise KeyError(f"Firm DataFrame missing columns: {sorted(missing)}")
    work = _ensure_numeric_t(df_firm)
    parent_branch = _parent_branch_value(work["branch"])
    parent = work[work["branch"] == parent_branch].copy()
    if parent.empty:
        raise ValueError("No parent observations found in firm DataFrame.")
    out = (
        parent.groupby(["path", "t"], as_index=False)
        .agg(
            b_mean_t=("b", "mean"),
            b_std_t=("b", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
            z_mean_t=("z", "mean"),
            z_std_t=("z", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
            n_parent_firms=("b", "size"),
        )
    )
    out["b_std_t"] = out["b_std_t"].fillna(0.0)
    out["z_std_t"] = out["z_std_t"].fillna(0.0)
    return out


def build_probe_table(df_firm: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    macro = _ensure_numeric_t(df_macro)
    pair = build_sdf_pairs_from_macro_ts(macro.copy(), include_hatc_lnk_t1=True)
    if pair.empty:
        raise ValueError("Macro pair table is empty.")
    pair["t_parent"] = pd.to_numeric(pair["t"], errors="coerce") - 1
    summary = build_parent_summaries(df_firm).rename(columns={"t": "t_parent"})
    pair = pair.merge(summary, on=["path", "t_parent"], how="left")
    required = [
        "x_t",
        "x_t1",
        "Hatc_t",
        "LnK_t",
        "Hatc_t1",
        "LnK_t1",
        "b_mean_t",
        "b_std_t",
        "z_mean_t",
        "z_std_t",
    ]
    return pair.dropna(subset=required).reset_index(drop=True)


def _path_split(paths: np.ndarray, seed: int, val_frac: float, test_frac: float) -> Dict[str, np.ndarray]:
    uniq = np.unique(paths.astype(np.int64))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n = perm.shape[0]
    n_test = min(max(1, int(round(n * test_frac))), max(1, n - 2))
    n_val = min(max(1, int(round(n * val_frac))), max(1, n - n_test - 1))
    test = perm[:n_test]
    val = perm[n_test:n_test + n_val]
    train = perm[n_test + n_val:]
    return {"train": train, "val": val, "test": test}


def _normalize(train_ref: np.ndarray, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_ref.mean(axis=0, keepdims=True)
    std = train_ref.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (arr - mean) / std, mean, std


@dataclass
class ProbeResult:
    name: str
    best_val_loss: float
    pred_test_norm: np.ndarray
    y_test_norm: np.ndarray
    train_history: List[float]
    val_history: List[float]


def _parse_hidden_dims(spec: str) -> List[int]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [int(p) for p in parts] if parts else [64, 64]


def train_probe(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    hidden_dims: Sequence[int],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
) -> ProbeResult:
    torch.manual_seed(seed)
    model = MLP(
        input_dim=x_train.shape[1],
        hidden_dims=list(hidden_dims),
        output_dim=y_train.shape[1],
        activation="gelu",
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    ds_train = TensorDataset(
        torch.from_numpy(x_train).to(torch.float32),
        torch.from_numpy(y_train).to(torch.float32),
    )
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    x_val_t = torch.from_numpy(x_val).to(device=device, dtype=torch.float32)
    y_val_t = torch.from_numpy(y_val).to(device=device, dtype=torch.float32)
    x_test_t = torch.from_numpy(x_test).to(device=device, dtype=torch.float32)

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    train_hist: List[float] = []
    val_hist: List[float] = []

    for epoch in range(epochs):
        model.train()
        total = 0
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            running += float(loss.item()) * xb.shape[0]
            total += xb.shape[0]
        train_loss = running / max(1, total)
        train_hist.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_t), y_val_t).item())
        val_hist.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_test = model(x_test_t).cpu().numpy()

    return ProbeResult(
        name=name,
        best_val_loss=float(best_val),
        pred_test_norm=pred_test,
        y_test_norm=y_test,
        train_history=train_hist,
        val_history=val_hist,
    )


def _plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(5.6, 5.0))
    plt.scatter(y_true, y_pred, s=8, alpha=0.25)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.any():
        lo = float(min(np.min(y_true[finite]), np.min(y_pred[finite])))
        hi = float(max(np.max(y_true[finite]), np.max(y_pred[finite])))
        plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_x_response(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    ylabel: str,
    path: Path,
    bins: int = 12,
) -> None:
    finite = np.isfinite(x) & np.isfinite(y_true) & np.isfinite(y_pred_a) & np.isfinite(y_pred_b)
    x = x[finite]
    y_true = y_true[finite]
    y_pred_a = y_pred_a[finite]
    y_pred_b = y_pred_b[finite]
    if x.size == 0:
        return
    edges = np.linspace(float(np.min(x)), float(np.max(x)), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mt, ma, mb = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x <= hi if hi == edges[-1] else x < hi)
        mt.append(float(np.mean(y_true[mask])) if mask.any() else float("nan"))
        ma.append(float(np.mean(y_pred_a[mask])) if mask.any() else float("nan"))
        mb.append(float(np.mean(y_pred_b[mask])) if mask.any() else float("nan"))
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(centers, mt, label="true", color="tab:blue", linewidth=2.0)
    plt.plot(centers, ma, label="baseline", color="tab:orange", linestyle="--", linewidth=2.0)
    plt.plot(centers, mb, label="augmented", color="tab:green", linestyle=":", linewidth=2.0)
    plt.xlabel("x_{t+1}")
    plt.ylabel(ylabel)
    plt.title(f"x response: {ylabel}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_loss_curves(results: Iterable[ProbeResult], path: Path) -> None:
    plt.figure(figsize=(6.4, 4.2))
    for result in results:
        plt.plot(result.train_history, label=f"{result.name} train")
        plt.plot(result.val_history, linestyle="--", label=f"{result.name} val")
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.title("Probe loss curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone FC1 state-sufficiency probe")
    parser.add_argument("--firm-pkl", type=Path, required=True)
    parser.add_argument("--macro-pkl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dims", type=str, default="64,64")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.macro_pkl.resolve().parents[2] / "experiments" / "fc1_state_probe"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_firm = pd.read_pickle(args.firm_pkl)
    df_macro = pd.read_pickle(args.macro_pkl)
    probe = build_probe_table(df_firm, df_macro)
    if probe.empty:
        raise ValueError("Probe dataset is empty after merge/filter.")

    split = _path_split(probe["path"].to_numpy(), args.seed, args.val_frac, args.test_frac)
    train_df = probe[probe["path"].isin(split["train"])].reset_index(drop=True)
    val_df = probe[probe["path"].isin(split["val"])].reset_index(drop=True)
    test_df = probe[probe["path"].isin(split["test"])].reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train/val/test split produced an empty partition.")

    baseline_cols = ["x_t", "x_t1", "Hatc_t", "LnK_t"]
    summary_cols = ["b_mean_t", "b_std_t", "z_mean_t", "z_std_t"]
    aug_cols = baseline_cols + summary_cols
    target_cols = ["Hatc_t1", "LnK_t1"]

    x_train_b = train_df[baseline_cols].to_numpy(dtype=np.float32)
    x_val_b = val_df[baseline_cols].to_numpy(dtype=np.float32)
    x_test_b = test_df[baseline_cols].to_numpy(dtype=np.float32)
    x_train_a = train_df[aug_cols].to_numpy(dtype=np.float32)
    x_val_a = val_df[aug_cols].to_numpy(dtype=np.float32)
    x_test_a = test_df[aug_cols].to_numpy(dtype=np.float32)
    y_train = train_df[target_cols].to_numpy(dtype=np.float32)
    y_val = val_df[target_cols].to_numpy(dtype=np.float32)
    y_test = test_df[target_cols].to_numpy(dtype=np.float32)

    x_train_b_n, _, _ = _normalize(x_train_b, x_train_b)
    x_val_b_n, _, _ = _normalize(x_train_b, x_val_b)
    x_test_b_n, _, _ = _normalize(x_train_b, x_test_b)
    x_train_a_n, _, _ = _normalize(x_train_a, x_train_a)
    x_val_a_n, _, _ = _normalize(x_train_a, x_val_a)
    x_test_a_n, _, _ = _normalize(x_train_a, x_test_a)
    y_train_n, y_mean, y_std = _normalize(y_train, y_train)
    y_val_n, _, _ = _normalize(y_train, y_val)
    y_test_n, _, _ = _normalize(y_train, y_test)

    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    device = torch.device(args.device)

    baseline = train_probe(
        "baseline",
        x_train_b_n,
        y_train_n,
        x_val_b_n,
        y_val_n,
        x_test_b_n,
        y_test_n,
        device,
        hidden_dims,
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.patience,
        args.seed,
    )
    augmented = train_probe(
        "augmented",
        x_train_a_n,
        y_train_n,
        x_val_a_n,
        y_val_n,
        x_test_a_n,
        y_test_n,
        device,
        hidden_dims,
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.patience,
        args.seed + 1,
    )

    baseline_pred = baseline.pred_test_norm * y_std + y_mean
    augmented_pred = augmented.pred_test_norm * y_std + y_mean
    y_test_phys = baseline.y_test_norm * y_std + y_mean

    summary = {
        "firm_pkl": str(args.firm_pkl),
        "macro_pkl": str(args.macro_pkl),
        "n_rows_total": int(probe.shape[0]),
        "n_rows_train": int(train_df.shape[0]),
        "n_rows_val": int(val_df.shape[0]),
        "n_rows_test": int(test_df.shape[0]),
        "n_paths_train": int(split["train"].shape[0]),
        "n_paths_val": int(split["val"].shape[0]),
        "n_paths_test": int(split["test"].shape[0]),
        "baseline": {
            **_prefixed_stats("hatc", y_test_phys[:, 0], baseline_pred[:, 0]),
            **_prefixed_stats("lnk", y_test_phys[:, 1], baseline_pred[:, 1]),
            "best_val_loss_norm": baseline.best_val_loss,
        },
        "augmented": {
            **_prefixed_stats("hatc", y_test_phys[:, 0], augmented_pred[:, 0]),
            **_prefixed_stats("lnk", y_test_phys[:, 1], augmented_pred[:, 1]),
            "best_val_loss_norm": augmented.best_val_loss,
        },
    }
    summary["delta_hatc_corr"] = summary["augmented"]["hatc_corr"] - summary["baseline"]["hatc_corr"]
    summary["delta_hatc_slope"] = summary["augmented"]["hatc_slope"] - summary["baseline"]["hatc_slope"]
    summary["delta_hatc_std_ratio"] = summary["augmented"]["hatc_std_ratio"] - summary["baseline"]["hatc_std_ratio"]
    summary["delta_hatc_r2"] = summary["augmented"]["hatc_r2"] - summary["baseline"]["hatc_r2"]
    summary["delta_lnk_corr"] = summary["augmented"]["lnk_corr"] - summary["baseline"]["lnk_corr"]
    summary["delta_lnk_slope"] = summary["augmented"]["lnk_slope"] - summary["baseline"]["lnk_slope"]
    summary["delta_lnk_std_ratio"] = summary["augmented"]["lnk_std_ratio"] - summary["baseline"]["lnk_std_ratio"]
    summary["delta_lnk_r2"] = summary["augmented"]["lnk_r2"] - summary["baseline"]["lnk_r2"]

    with open(args.output_dir / "probe_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    probe.to_pickle(args.output_dir / "probe_dataset.pkl")
    pd.DataFrame(
        {
            "path": test_df["path"].to_numpy(dtype=int),
            "branch": test_df["branch"].to_numpy(dtype=int),
            "t_child": test_df["t"].to_numpy(dtype=int),
            "x_t": test_df["x_t"].to_numpy(dtype=float),
            "x_t1": test_df["x_t1"].to_numpy(dtype=float),
            "Hatc_true": y_test_phys[:, 0],
            "LnK_true": y_test_phys[:, 1],
            "Hatc_pred_baseline": baseline_pred[:, 0],
            "LnK_pred_baseline": baseline_pred[:, 1],
            "Hatc_pred_augmented": augmented_pred[:, 0],
            "LnK_pred_augmented": augmented_pred[:, 1],
            "b_mean_t": test_df["b_mean_t"].to_numpy(dtype=float),
            "b_std_t": test_df["b_std_t"].to_numpy(dtype=float),
            "z_mean_t": test_df["z_mean_t"].to_numpy(dtype=float),
            "z_std_t": test_df["z_std_t"].to_numpy(dtype=float),
        }
    ).to_csv(args.output_dir / "probe_test_predictions.csv", index=False)

    _plot_loss_curves([baseline, augmented], args.output_dir / "probe_loss_curves.png")
    _plot_scatter(y_test_phys[:, 0], baseline_pred[:, 0], "Baseline probe: Hatc", args.output_dir / "probe_hatc_scatter_baseline.png")
    _plot_scatter(y_test_phys[:, 0], augmented_pred[:, 0], "Augmented probe: Hatc", args.output_dir / "probe_hatc_scatter_augmented.png")
    _plot_scatter(y_test_phys[:, 1], baseline_pred[:, 1], "Baseline probe: LnK", args.output_dir / "probe_lnk_scatter_baseline.png")
    _plot_scatter(y_test_phys[:, 1], augmented_pred[:, 1], "Augmented probe: LnK", args.output_dir / "probe_lnk_scatter_augmented.png")
    _plot_x_response(test_df["x_t1"].to_numpy(dtype=float), y_test_phys[:, 0], baseline_pred[:, 0], augmented_pred[:, 0], "Hatc", args.output_dir / "probe_hatc_vs_x.png")
    _plot_x_response(test_df["x_t1"].to_numpy(dtype=float), y_test_phys[:, 1], baseline_pred[:, 1], augmented_pred[:, 1], "LnK", args.output_dir / "probe_lnk_vs_x.png")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
