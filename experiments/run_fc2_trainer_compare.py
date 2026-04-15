"""
Compare FC2 supervised trainers on the same simulated episode dataset.

Goal:
- hold the simulated dataset fixed
- compare the standalone probe trainer against an episode-style trainer
- isolate whether the hatc gap comes from training protocol rather than data
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config, HyperParams  # noqa: E402
from models.fc2 import FC2HatcModel, FC2LnkModel  # noqa: E402
from training.gradient_utils import gradient_protection  # noqa: E402
from training.scheduler import LearningRateScheduler  # noqa: E402
from experiments.run_fc2_supervised_probe import (  # noqa: E402
    _eval_poly_baseline_np,
    _fit_poly_baseline_np,
    _normalize,
    _parse_hidden_dims,
    _path_split,
    _plot_scatter,
    _prefixed_stats,
    build_supervised_probe_dataset,
    simulate_probe_data,
    train_probe,
)


@dataclass
class TrainerResult:
    name: str
    pred_test: np.ndarray
    y_test: np.ndarray
    train_history: List[float]
    val_history: List[float]
    best_val_loss: float


def _plot_loss_curves(results: Sequence[TrainerResult], path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.4, 4.2))
    for result in results:
        plt.plot(result.train_history, label=f"{result.name} train")
        plt.plot(result.val_history, linestyle="--", label=f"{result.name} val")
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.title("FC2 trainer loss curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _tensor_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x).to(torch.float32),
        torch.from_numpy(y).to(torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_probe_lite_scalar(
    name: str,
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    patience: int,
    normalize_y_loss: bool,
    fit_hatc_baseline: bool,
) -> TrainerResult:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    train_loader = _tensor_loader(x_train, y_train, batch_size=batch_size, shuffle=True)

    x_train_t = torch.from_numpy(x_train).to(device=device, dtype=torch.float32)
    x_val_t = torch.from_numpy(x_val).to(device=device, dtype=torch.float32)
    y_val_t = torch.from_numpy(y_val).to(device=device, dtype=torch.float32)
    x_test_t = torch.from_numpy(x_test).to(device=device, dtype=torch.float32)

    if fit_hatc_baseline and hasattr(model, "fit_x_baseline"):
        model.fit_x_baseline(x_train_t.detach().cpu(), torch.from_numpy(y_train).to(torch.float32))

    def _baseline(phi: torch.Tensor) -> torch.Tensor:
        if fit_hatc_baseline and hasattr(model, "x_baseline"):
            return model.x_baseline(phi)
        return torch.zeros((phi.shape[0], 1), device=phi.device, dtype=phi.dtype)

    with torch.no_grad():
        if normalize_y_loss:
            train_target = torch.from_numpy(y_train).to(device=device, dtype=torch.float32)
            train_resid = train_target - _baseline(x_train_t)
            y_mean_t = train_resid.mean(dim=0, keepdim=True)
            y_std_t = train_resid.std(dim=0, keepdim=True).clamp(min=1e-6)
        else:
            y_mean_t = torch.zeros((1, y_train.shape[1]), device=device, dtype=torch.float32)
            y_std_t = torch.ones((1, y_train.shape[1]), device=device, dtype=torch.float32)

    train_hist: List[float] = []
    val_hist: List[float] = []
    best_val = float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(epochs):
        model.train()
        running = 0.0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            if normalize_y_loss:
                pred_loss = (pred - _baseline(xb) - y_mean_t) / y_std_t
                y_loss = (yb - _baseline(xb) - y_mean_t) / y_std_t
                loss = loss_fn(pred_loss, y_loss)
            else:
                loss = loss_fn(pred, yb)
            loss.backward()
            grad_norm, had_nan = gradient_protection(
                model.parameters(),
                max_norm=max_grad_norm,
            )
            if had_nan:
                raise RuntimeError(f"NaN gradient detected in trainer: {name}")
            optimizer.step()
            running += float(loss.item()) * xb.shape[0]
            total += xb.shape[0]
        train_loss = running / max(1, total)
        train_hist.append(train_loss)

        model.eval()
        with torch.no_grad():
            pred_val = model(x_val_t)
            if normalize_y_loss:
                pred_val_loss = (pred_val - _baseline(x_val_t) - y_mean_t) / y_std_t
                y_val_loss = (y_val_t - _baseline(x_val_t) - y_mean_t) / y_std_t
                val_loss = float(loss_fn(pred_val_loss, y_val_loss).item())
            else:
                val_loss = float(loss_fn(pred_val, y_val_t).item())
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

    return TrainerResult(
        name=name,
        pred_test=pred_test,
        y_test=y_test,
        train_history=train_hist,
        val_history=val_hist,
        best_val_loss=best_val,
    )


def _episode_style_models(
    hidden_dims: Sequence[int],
    hatc_dropout: float,
    lnk_dropout: float,
) -> tuple[FC2HatcModel, FC2LnkModel]:
    hatc_model = FC2HatcModel(
        input_dim=Config.FC2_INPUT_DIM,
        hidden_dims=list(hidden_dims),
        quantile_num=Config.QUANTILE_NUM,
        dropout=hatc_dropout,
    )
    lnk_model = FC2LnkModel(
        input_dim=Config.FC2_INPUT_DIM + Config.QUANTILE_NUM,
        hidden_dims=list(hidden_dims),
        quantile_num=Config.QUANTILE_NUM,
        dropout=lnk_dropout,
    )
    return hatc_model, lnk_model


def _trainer_summary(
    result: TrainerResult,
    target_name: str,
) -> Dict[str, float]:
    summary = {"best_val_loss_raw": float(result.best_val_loss)}
    summary.update(_prefixed_stats(target_name, result.y_test[:, 0], result.pred_test[:, 0]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FC2 probe trainer vs episode-style trainer on the same dataset")
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--ckpt-prefix", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-paths", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=Config.SIMULATE_GROUP_SIZE)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--fc2-as-main-macro-state", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hidden-dims", type=str, default="128,64,32")
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-batch-size", type=int, default=4096)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-patience", type=int, default=30)
    parser.add_argument("--probe-lite-epochs", type=int, default=HyperParams().fc2_supervised_pretrain_epochs)
    parser.add_argument("--probe-lite-batch-size", type=int, default=HyperParams().fc2_pretrain_batch_size)
    parser.add_argument("--probe-lite-lr", type=float, default=HyperParams().fc2_pretrain_lr)
    parser.add_argument("--probe-lite-weight-decay", type=float, default=HyperParams().fc2_pretrain_weight_decay)
    parser.add_argument("--probe-lite-patience", type=int, default=HyperParams().fc2_pretrain_patience)
    parser.add_argument("--episode-max-grad-norm", type=float, default=HyperParams().fc2_max_grad_norm)
    parser.add_argument("--probe-lite-hatc-dropout", type=float, default=0.0)
    parser.add_argument("--probe-lite-lnk-dropout", type=float, default=0.0)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = ROOT / "experiments" / "fc2_trainer_compare"
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
    probe.to_pickle(args.output_dir / "comparison_dataset.pkl")

    hatc_feature_cols = [f"hatc_phi_{k}" for k in range(Config.FC2_INPUT_DIM)]
    lnk_feature_cols = [f"lnk_phi_{k}" for k in range(Config.FC2_INPUT_DIM + Config.QUANTILE_NUM)]

    split = _path_split(probe["path"].to_numpy(), args.seed, args.val_frac, args.test_frac)
    train_df = probe[probe["path"].isin(split["train"])].reset_index(drop=True)
    val_df = probe[probe["path"].isin(split["val"])].reset_index(drop=True)
    test_df = probe[probe["path"].isin(split["test"])].reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train/val/test split produced an empty partition.")

    hidden_dims = _parse_hidden_dims(args.hidden_dims)

    # Probe trainer: identical protocol to run_fc2_supervised_probe.py
    probe_results_summary: Dict[str, Dict[str, float]] = {}
    probe_predictions: Dict[str, np.ndarray] = {}
    probe_results = []
    for task_name, feature_cols, target_col in (
        ("hatc", hatc_feature_cols, "hatc"),
        ("lnk", lnk_feature_cols, "lnk"),
    ):
        x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
        x_val = val_df[feature_cols].to_numpy(dtype=np.float32)
        x_test = test_df[feature_cols].to_numpy(dtype=np.float32)
        x_train_n, _, _ = _normalize(x_train, x_train)
        x_val_n, _, _ = _normalize(x_train, x_val)
        x_test_n, _, _ = _normalize(x_train, x_test)
        y_train = train_df[[target_col]].to_numpy(dtype=np.float32)
        y_val = val_df[[target_col]].to_numpy(dtype=np.float32)
        y_test = test_df[[target_col]].to_numpy(dtype=np.float32)
        y_train_model = y_train
        y_val_model = y_val
        y_test_model = y_test
        baseline_coef = None
        baseline_x_mean = 0.0
        baseline_x_scale = 1.0
        baseline_test = None
        if task_name == "hatc":
            baseline_coef, baseline_x_mean, baseline_x_scale = _fit_poly_baseline_np(
                x_train[:, -1], y_train, degree=2
            )
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
            f"probe_{task_name}",
            x_train_n,
            y_train_n,
            x_val_n,
            y_val_n,
            x_test_n,
            y_test_n,
            device,
            hidden_dims,
            args.probe_epochs,
            args.probe_batch_size,
            args.probe_lr,
            args.probe_weight_decay,
            args.probe_patience,
            args.seed + len(probe_results),
        )
        probe_results.append(result)
        pred_phys = result.pred_test_norm * y_std + y_mean
        true_phys = result.y_test_norm * y_std + y_mean
        if baseline_test is not None:
            pred_phys = pred_phys + baseline_test
            true_phys = y_test
        probe_predictions[task_name] = pred_phys
        task_summary: Dict[str, float] = {"best_val_loss_norm": result.best_val_loss}
        if baseline_coef is not None:
            for idx, val in enumerate(baseline_coef.reshape(-1).tolist()):
                task_summary[f"x_baseline_coef_{idx}"] = float(val)
            task_summary["x_baseline_mean"] = float(baseline_x_mean)
            task_summary["x_baseline_scale"] = float(baseline_x_scale)
        task_summary.update(_prefixed_stats(task_name, true_phys[:, 0], pred_phys[:, 0]))
        probe_results_summary[task_name] = task_summary

    # Probe-lite trainer: matches current mainline supervised pretrain more closely.
    x_hatc_train = train_df[hatc_feature_cols].to_numpy(dtype=np.float32)
    x_hatc_val = val_df[hatc_feature_cols].to_numpy(dtype=np.float32)
    x_hatc_test = test_df[hatc_feature_cols].to_numpy(dtype=np.float32)
    y_hatc_train = train_df[["hatc"]].to_numpy(dtype=np.float32)
    y_hatc_val = val_df[["hatc"]].to_numpy(dtype=np.float32)
    y_hatc_test = test_df[["hatc"]].to_numpy(dtype=np.float32)

    x_lnk_train = train_df[lnk_feature_cols].to_numpy(dtype=np.float32)
    x_lnk_val = val_df[lnk_feature_cols].to_numpy(dtype=np.float32)
    x_lnk_test = test_df[lnk_feature_cols].to_numpy(dtype=np.float32)
    y_lnk_train = train_df[["lnk"]].to_numpy(dtype=np.float32)
    y_lnk_val = val_df[["lnk"]].to_numpy(dtype=np.float32)
    y_lnk_test = test_df[["lnk"]].to_numpy(dtype=np.float32)

    probe_lite_hatc_raw, probe_lite_lnk_raw = _episode_style_models(
        hidden_dims=hidden_dims,
        hatc_dropout=args.probe_lite_hatc_dropout,
        lnk_dropout=args.probe_lite_lnk_dropout,
    )
    probe_lite_hatc_ynorm, probe_lite_lnk_ynorm = _episode_style_models(
        hidden_dims=hidden_dims,
        hatc_dropout=args.probe_lite_hatc_dropout,
        lnk_dropout=args.probe_lite_lnk_dropout,
    )

    probe_lite_hatc_raw_res = train_probe_lite_scalar(
        name="probe_lite_hatc_raw",
        model=probe_lite_hatc_raw,
        x_train=x_hatc_train,
        y_train=y_hatc_train,
        x_val=x_hatc_val,
        y_val=y_hatc_val,
        x_test=x_hatc_test,
        y_test=y_hatc_test,
        device=device,
        epochs=args.probe_lite_epochs,
        batch_size=args.probe_lite_batch_size,
        lr=args.probe_lite_lr,
        weight_decay=args.probe_lite_weight_decay,
        max_grad_norm=args.episode_max_grad_norm,
        patience=args.probe_lite_patience,
        normalize_y_loss=False,
        fit_hatc_baseline=True,
    )
    probe_lite_hatc_ynorm_res = train_probe_lite_scalar(
        name="probe_lite_hatc_y_norm",
        model=probe_lite_hatc_ynorm,
        x_train=x_hatc_train,
        y_train=y_hatc_train,
        x_val=x_hatc_val,
        y_val=y_hatc_val,
        x_test=x_hatc_test,
        y_test=y_hatc_test,
        device=device,
        epochs=args.probe_lite_epochs,
        batch_size=args.probe_lite_batch_size,
        lr=args.probe_lite_lr,
        weight_decay=args.probe_lite_weight_decay,
        max_grad_norm=args.episode_max_grad_norm,
        patience=args.probe_lite_patience,
        normalize_y_loss=True,
        fit_hatc_baseline=True,
    )
    probe_lite_lnk_raw_res = train_probe_lite_scalar(
        name="probe_lite_lnk_raw",
        model=probe_lite_lnk_raw,
        x_train=x_lnk_train,
        y_train=y_lnk_train,
        x_val=x_lnk_val,
        y_val=y_lnk_val,
        x_test=x_lnk_test,
        y_test=y_lnk_test,
        device=device,
        epochs=args.probe_lite_epochs,
        batch_size=args.probe_lite_batch_size,
        lr=args.probe_lite_lr,
        weight_decay=args.probe_lite_weight_decay,
        max_grad_norm=args.episode_max_grad_norm,
        patience=args.probe_lite_patience,
        normalize_y_loss=False,
        fit_hatc_baseline=False,
    )

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
        "probe_full_trainer": probe_results_summary,
        "probe_lite_raw_trainer": {
            "hatc": _trainer_summary(probe_lite_hatc_raw_res, "hatc"),
            "lnk": _trainer_summary(probe_lite_lnk_raw_res, "lnk"),
        },
        "probe_lite_hatc_y_norm_trainer": {
            "hatc": _trainer_summary(probe_lite_hatc_ynorm_res, "hatc"),
            "lnk": _trainer_summary(probe_lite_lnk_raw_res, "lnk"),
        },
        "probe_lite_config": {
            "epochs": int(args.probe_lite_epochs),
            "batch_size": int(args.probe_lite_batch_size),
            "lr": float(args.probe_lite_lr),
            "weight_decay": float(args.probe_lite_weight_decay),
            "patience": int(args.probe_lite_patience),
            "max_grad_norm": float(args.episode_max_grad_norm),
            "hatc_dropout": float(args.probe_lite_hatc_dropout),
            "lnk_dropout": float(args.probe_lite_lnk_dropout),
            "hatc_y_norm_loss": True,
        },
        "probe_config": {
            "epochs": int(args.probe_epochs),
            "batch_size": int(args.probe_batch_size),
            "lr": float(args.probe_lr),
            "weight_decay": float(args.probe_weight_decay),
            "patience": int(args.probe_patience),
        },
    }

    with open(args.output_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pred_frame = pd.DataFrame(
        {
            "path": test_df["path"].to_numpy(dtype=int),
            "node_type": test_df["node_type"].to_numpy(dtype=int),
            "branch": test_df["branch"].to_numpy(dtype=int),
            "hatc_true": test_df["hatc"].to_numpy(dtype=float),
            "lnk_true": test_df["lnk"].to_numpy(dtype=float),
            "hatc_pred_probe_full": probe_predictions["hatc"][:, 0],
            "lnk_pred_probe_full": probe_predictions["lnk"][:, 0],
            "hatc_pred_probe_lite_raw": probe_lite_hatc_raw_res.pred_test[:, 0],
            "hatc_pred_probe_lite_hatc_y_norm": probe_lite_hatc_ynorm_res.pred_test[:, 0],
            "lnk_pred_probe_lite_raw": probe_lite_lnk_raw_res.pred_test[:, 0],
        }
    )
    pred_frame.to_csv(args.output_dir / "comparison_test_predictions.csv", index=False)

    _plot_scatter(
        pred_frame["hatc_true"].to_numpy(dtype=float),
        pred_frame["hatc_pred_probe_full"].to_numpy(dtype=float),
        "FC2 compare: probe full trainer hatc",
        args.output_dir / "compare_probe_full_hatc_scatter.png",
    )
    _plot_scatter(
        pred_frame["hatc_true"].to_numpy(dtype=float),
        pred_frame["hatc_pred_probe_lite_raw"].to_numpy(dtype=float),
        "FC2 compare: probe-lite raw hatc",
        args.output_dir / "compare_probe_lite_raw_hatc_scatter.png",
    )
    _plot_scatter(
        pred_frame["hatc_true"].to_numpy(dtype=float),
        pred_frame["hatc_pred_probe_lite_hatc_y_norm"].to_numpy(dtype=float),
        "FC2 compare: probe-lite hatc y-norm",
        args.output_dir / "compare_probe_lite_hatc_y_norm_scatter.png",
    )
    _plot_scatter(
        pred_frame["lnk_true"].to_numpy(dtype=float),
        pred_frame["lnk_pred_probe_full"].to_numpy(dtype=float),
        "FC2 compare: probe full trainer lnk",
        args.output_dir / "compare_probe_full_lnk_scatter.png",
    )
    _plot_scatter(
        pred_frame["lnk_true"].to_numpy(dtype=float),
        pred_frame["lnk_pred_probe_lite_raw"].to_numpy(dtype=float),
        "FC2 compare: probe-lite raw lnk",
        args.output_dir / "compare_probe_lite_raw_lnk_scatter.png",
    )
    _plot_loss_curves(
        [probe_lite_hatc_raw_res, probe_lite_hatc_ynorm_res, probe_lite_lnk_raw_res],
        args.output_dir / "compare_probe_lite_loss_curves.png",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
