"""
Standalone FC2 supervised probe.

Goal:
- bypass policy/value closure training entirely
 - test whether the current split FC2 summaries contain enough information to
   predict node-level macro targets

Tasks:
1. hatc-only
2. lnk-only
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

from config import Config  # noqa: E402
from data.simulate_ts import SimulateTS  # noqa: E402
from experiments.run_utils import build_models  # noqa: E402
from losses.FC2losspipe import FC2Pipeline  # noqa: E402
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


def _normalize(train_ref: np.ndarray, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_ref.mean(axis=0, keepdims=True)
    std = train_ref.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (arr - mean) / std, mean, std


def _fit_poly_baseline_np(
    x_train: np.ndarray,
    y_train: np.ndarray,
    degree: int = 2,
) -> tuple[np.ndarray, float, float]:
    x_vec = np.asarray(x_train, dtype=np.float64).reshape(-1, 1)
    y_vec = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
    mask = np.isfinite(x_vec).reshape(-1) & np.isfinite(y_vec).reshape(-1)
    if mask.sum() < degree + 1:
        return np.zeros((degree + 1, 1), dtype=np.float64), 0.0, 1.0
    x_fit = x_vec[mask]
    x_mean = float(x_fit.mean())
    x_scale = float(max(x_fit.std(), 1e-6))
    x_std = (x_fit - x_mean) / x_scale
    cols = [np.ones((int(mask.sum()), 1), dtype=np.float64)]
    for d in range(1, degree + 1):
        cols.append(x_std ** d)
    design = np.concatenate(cols, axis=1)
    coef, *_ = np.linalg.lstsq(design, y_vec[mask], rcond=None)
    return coef, x_mean, x_scale


def _eval_poly_baseline_np(x: np.ndarray, coef: np.ndarray, x_mean: float, x_scale: float) -> np.ndarray:
    x_vec = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    degree = int(coef.shape[0] - 1)
    x_std = (x_vec - x_mean) / max(x_scale, 1e-6)
    cols = [np.ones_like(x_vec)]
    for d in range(1, degree + 1):
        cols.append(x_std ** d)
    design = np.concatenate(cols, axis=1)
    return design @ coef


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
    return [int(p) for p in parts] if parts else list(Config.FC2_HIDDEN_DIMS)


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
        running = 0.0
        total = 0
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


def _plot_loss_curves(results: Iterable[ProbeResult], path: Path) -> None:
    plt.figure(figsize=(6.4, 4.2))
    for result in results:
        plt.plot(result.train_history, label=f"{result.name} train")
        plt.plot(result.val_history, linestyle="--", label=f"{result.name} val")
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.title("FC2 supervised probe loss curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def build_supervised_probe_dataset(
    pipe: FC2Pipeline,
) -> pd.DataFrame:
    parent_x = {
        k: v.detach().cpu()
        for k, v in pipe.build_fc2_input_parent_dual().items()
    }
    child_x = {
        k: v.detach().cpu()
        for k, v in pipe.build_fc2_input_children_dual(pipe.child_states_list, pipe.child_present_list).items()
    }
    parent_y = pipe.build_supervised_targets_parent().detach().cpu()
    child_y = pipe.build_supervised_targets_children().detach().cpu()

    records: List[Dict[str, float]] = []
    for local_idx, path_val in enumerate(pipe.path_values):
        hatc_feat = parent_x["hatc"][local_idx]
        lnk_feat = parent_x["lnk"][local_idx]
        target = parent_y[local_idx]
        records.append(
            {
                "path": float(path_val),
                "node_type": 0.0,
                "branch": -1.0,
                "hatc": float(target[1].item()),
                "lnk": float(target[0].item()),
                **{f"hatc_phi_{k}": float(hatc_feat[k].item()) for k in range(hatc_feat.shape[0])},
                **{f"lnk_phi_{k}": float(lnk_feat[k].item()) for k in range(lnk_feat.shape[0])},
            }
        )
        for branch_idx in range(pipe.branch_num):
            if not pipe.child_present_list[local_idx][:, branch_idx].any():
                continue
            hatc_feat = child_x["hatc"][local_idx, branch_idx]
            lnk_feat = child_x["lnk"][local_idx, branch_idx]
            target = child_y[local_idx, branch_idx]
            records.append(
                {
                    "path": float(path_val),
                    "node_type": 1.0,
                    "branch": float(branch_idx),
                    "hatc": float(target[1].item()),
                    "lnk": float(target[0].item()),
                    **{f"hatc_phi_{k}": float(hatc_feat[k].item()) for k in range(hatc_feat.shape[0])},
                    **{f"lnk_phi_{k}": float(lnk_feat[k].item()) for k in range(lnk_feat.shape[0])},
                }
            )

    if not records:
        raise ValueError("No supervised FC2 probe records were built.")
    return pd.DataFrame.from_records(records)


def simulate_probe_data(
    device: torch.device,
    ckpt_dir: Path | None,
    ckpt_prefix: str | None,
    n_paths: int,
    group_size: int,
    horizon: int,
    fc2_as_main_macro_state: bool,
) -> FC2Pipeline:
    models = build_models(device=device, ckpt_dir=ckpt_dir, ckpt_prefix=ckpt_prefix)
    simulator = SimulateTS(
        models=models,
        config=Config,
        n_paths=n_paths,
        group_size=group_size,
        horizon=horizon,
        branch_num=Config.BRANCH_NUM,
        fc2_as_main_macro_state=fc2_as_main_macro_state,
        device=device,
    )
    out = simulator.simulate_tensor()
    return FC2Pipeline(
        firm_table=out.firm,
        macro_table=out.macro,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone FC2 supervised probe")
    parser.add_argument("--ckpt-dir", type=Path, default=None, help="Checkpoint directory used to build simulation models.")
    parser.add_argument("--ckpt-prefix", type=str, default=None, help="Optional checkpoint prefix, e.g. ep0 or ep3.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-paths", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=Config.SIMULATE_GROUP_SIZE)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--fc2-as-main-macro-state", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hidden-dims", type=str, default="128,64,32")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = ROOT / "experiments" / "fc2_supervised_probe"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    pipe = simulate_probe_data(
        device=device,
        ckpt_dir=args.ckpt_dir,
        ckpt_prefix=args.ckpt_prefix,
        n_paths=args.n_paths,
        group_size=args.group_size,
        horizon=args.horizon,
        fc2_as_main_macro_state=bool(args.fc2_as_main_macro_state),
    )

    probe = build_supervised_probe_dataset(pipe)
    hatc_feature_cols = [f"hatc_phi_{k}" for k in range(Config.FC2_INPUT_DIM)]
    lnk_feature_cols = [f"lnk_phi_{k}" for k in range(Config.FC2_INPUT_DIM + Config.QUANTILE_NUM)]
    task_specs = {
        "hatc_only": {
            "targets": ["hatc"],
            "features": hatc_feature_cols,
        },
        "lnk_only": {
            "targets": ["lnk"],
            "features": lnk_feature_cols,
        },
    }

    split = _path_split(probe["path"].to_numpy(), args.seed, args.val_frac, args.test_frac)
    train_df = probe[probe["path"].isin(split["train"])].reset_index(drop=True)
    val_df = probe[probe["path"].isin(split["val"])].reset_index(drop=True)
    test_df = probe[probe["path"].isin(split["test"])].reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train/val/test split produced an empty partition.")

    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    results_summary: Dict[str, Dict[str, float]] = {}
    probe_results: List[ProbeResult] = []
    pred_cols: Dict[str, np.ndarray] = {}

    for task_name, task_spec in task_specs.items():
        feature_cols = task_spec["features"]
        target_cols = task_spec["targets"]
        x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
        x_val = val_df[feature_cols].to_numpy(dtype=np.float32)
        x_test = test_df[feature_cols].to_numpy(dtype=np.float32)
        x_train_n, _, _ = _normalize(x_train, x_train)
        x_val_n, _, _ = _normalize(x_train, x_val)
        x_test_n, _, _ = _normalize(x_train, x_test)
        y_train = train_df[target_cols].to_numpy(dtype=np.float32)
        y_val = val_df[target_cols].to_numpy(dtype=np.float32)
        y_test = test_df[target_cols].to_numpy(dtype=np.float32)
        y_train_model = y_train
        y_val_model = y_val
        y_test_model = y_test
        baseline_coef = None
        baseline_x_mean = 0.0
        baseline_x_scale = 1.0
        baseline_train = None
        baseline_val = None
        baseline_test = None
        if task_name == "hatc_only":
            baseline_coef, baseline_x_mean, baseline_x_scale = _fit_poly_baseline_np(x_train[:, -1], y_train, degree=2)
            baseline_train = _eval_poly_baseline_np(x_train[:, -1], baseline_coef, baseline_x_mean, baseline_x_scale).astype(np.float32)
            baseline_val = _eval_poly_baseline_np(x_val[:, -1], baseline_coef, baseline_x_mean, baseline_x_scale).astype(np.float32)
            baseline_test = _eval_poly_baseline_np(x_test[:, -1], baseline_coef, baseline_x_mean, baseline_x_scale).astype(np.float32)
            y_train_model = y_train - baseline_train
            y_val_model = y_val - baseline_val
            y_test_model = y_test - baseline_test
        y_train_n, y_mean, y_std = _normalize(y_train_model, y_train_model)
        y_val_n, _, _ = _normalize(y_train_model, y_val_model)
        y_test_n, _, _ = _normalize(y_train_model, y_test_model)

        result = train_probe(
            task_name,
            x_train_n,
            y_train_n,
            x_val_n,
            y_val_n,
            x_test_n,
            y_test_n,
            device,
            hidden_dims,
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            args.patience,
            args.seed + len(probe_results),
        )
        probe_results.append(result)
        pred_phys = result.pred_test_norm * y_std + y_mean
        true_phys = result.y_test_norm * y_std + y_mean
        if baseline_test is not None:
            pred_phys = pred_phys + baseline_test
            true_phys = y_test
        pred_cols[task_name] = pred_phys

        task_summary: Dict[str, float] = {"best_val_loss_norm": result.best_val_loss}
        if baseline_coef is not None:
            for idx, val in enumerate(baseline_coef.reshape(-1).tolist()):
                task_summary[f"x_baseline_coef_{idx}"] = float(val)
            task_summary["x_baseline_mean"] = float(baseline_x_mean)
            task_summary["x_baseline_scale"] = float(baseline_x_scale)
        for idx, col in enumerate(target_cols):
            task_summary.update(_prefixed_stats(col, true_phys[:, idx], pred_phys[:, idx]))
        results_summary[task_name] = task_summary

    summary = {
        "ckpt_dir": str(args.ckpt_dir) if args.ckpt_dir is not None else None,
        "ckpt_prefix": args.ckpt_prefix,
        "n_paths_simulated": int(args.n_paths),
        "group_size": int(args.group_size),
        "horizon": int(args.horizon),
        "fc2_as_main_macro_state": bool(args.fc2_as_main_macro_state),
        "n_rows_total": int(probe.shape[0]),
        "n_rows_train": int(train_df.shape[0]),
        "n_rows_val": int(val_df.shape[0]),
        "n_rows_test": int(test_df.shape[0]),
        "n_paths_train": int(split["train"].shape[0]),
        "n_paths_val": int(split["val"].shape[0]),
        "n_paths_test": int(split["test"].shape[0]),
        "tasks": results_summary,
    }

    with open(args.output_dir / "probe_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    probe.to_pickle(args.output_dir / "probe_dataset.pkl")
    pred_frame = pd.DataFrame(
        {
            "path": test_df["path"].to_numpy(dtype=int),
            "node_type": test_df["node_type"].to_numpy(dtype=int),
            "branch": test_df["branch"].to_numpy(dtype=int),
            "hatc_true": test_df["hatc"].to_numpy(dtype=float),
            "lnk_true": test_df["lnk"].to_numpy(dtype=float),
            "hatc_pred_hatc_only": pred_cols["hatc_only"][:, 0],
            "lnk_pred_lnk_only": pred_cols["lnk_only"][:, 0],
        }
    )
    pred_frame.to_csv(args.output_dir / "probe_test_predictions.csv", index=False)

    _plot_loss_curves(probe_results, args.output_dir / "probe_loss_curves.png")
    _plot_scatter(
        test_df["hatc"].to_numpy(dtype=float),
        pred_cols["hatc_only"][:, 0],
        "FC2 probe: hatc-only",
        args.output_dir / "probe_hatc_only_scatter.png",
    )
    _plot_scatter(
        test_df["lnk"].to_numpy(dtype=float),
        pred_cols["lnk_only"][:, 0],
        "FC2 probe: lnk-only",
        args.output_dir / "probe_lnk_only_scatter.png",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
