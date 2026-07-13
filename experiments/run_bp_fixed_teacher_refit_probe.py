from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from experiments.export_bp_deep_diagnostics import (  # noqa: E402
    make_episode_batches,
    select_exported_states,
    stable_logit_with_censoring,
)
from experiments.run_utils import build_hyperparams, build_models  # noqa: E402
from training.bp_policy_loss import compute_target_grid_policy_distillation_loss  # noqa: E402


TRAINABLE_PREFIXES = ("policy_encoder.", "bp0_head.", "bpi_head.")


def parse_record_steps(value: str) -> List[int]:
    steps = sorted({int(x) for x in value.split(",") if x.strip()})
    if not steps or steps[0] != 0:
        steps = [0] + steps
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-teacher BP policy refit probe.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--firm-pkl", type=Path, required=True)
    parser.add_argument("--decomposition-summary", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--record-steps",
        type=parse_record_steps,
        default=parse_record_steps("0,1,2,5,10,20,50,100,200,500,1000"),
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-branches", type=int, default=2)
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def configure_refit_trainable_parameters(model: torch.nn.Module) -> List[str]:
    trainable = []
    for name, param in model.named_parameters():
        allowed = name.startswith(TRAINABLE_PREFIXES)
        param.requires_grad_(allowed)
        if allowed:
            trainable.append(name)
    return trainable


def assert_only_allowed_changed(
    before: Dict[str, torch.Tensor],
    after_model: torch.nn.Module,
) -> None:
    for name, param in after_model.named_parameters():
        changed = not torch.equal(before[name].to(param.device), param.detach())
        allowed = name.startswith(TRAINABLE_PREFIXES)
        if changed and not allowed:
            raise AssertionError(f"Frozen parameter changed during refit probe: {name}")


def fixed_targets(summary_df: pd.DataFrame, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    targets = {}
    for branch, key in [("p0", "bp0"), ("pi", "bpI"), ("mix", "bp_mix")]:
        branch_df = summary_df.loc[summary_df["branch"] == branch].sort_values("state_pos")
        if branch_df.empty:
            raise RuntimeError(f"Missing decomposition summary rows for branch={branch}")
        targets[key] = torch.as_tensor(
            branch_df["bp_star_teacher"].to_numpy(dtype=np.float64),
            device=device,
            dtype=dtype,
        ).reshape(-1, 1)
        targets[f"{key}_confidence"] = torch.as_tensor(
            branch_df["confidence_weight"].to_numpy(dtype=np.float64),
            device=device,
            dtype=dtype,
        ).reshape(-1, 1)
        if branch == "mix":
            targets["bp_mix_survival_weight"] = torch.as_tensor(
                branch_df["mix_survival_weight"].to_numpy(dtype=np.float64),
                device=device,
                dtype=dtype,
            ).reshape(-1, 1)
    return targets


def split_indices(n: int, holdout_fraction: float, seed: int, device: torch.device) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator)
    n_holdout = int(round(n * float(holdout_fraction)))
    n_holdout = min(max(n_holdout, 1 if n > 1 else 0), max(n - 1, 0))
    holdout = perm[:n_holdout]
    train = perm[n_holdout:]
    if train.numel() == 0:
        train = holdout
        holdout = holdout[:0]
    return {
        "train": train.to(device=device, dtype=torch.long),
        "holdout": holdout.to(device=device, dtype=torch.long),
    }


def forward_policy_bundle(model: torch.nn.Module, parent_state: torch.Tensor) -> Dict[str, torch.Tensor]:
    out = model(parent_state)
    mix_weight = out.bar_i_cond.clamp(0.0, 1.0)
    bp_mix = (1.0 - mix_weight) * out.bp0 + mix_weight * out.bpI
    bp0_logit, _, _ = stable_logit_with_censoring(out.bp0)
    bpI_logit, _, _ = stable_logit_with_censoring(out.bpI)
    return {
        "bp0": out.bp0,
        "bpI": out.bpI,
        "bp_mix": bp_mix,
        "bp0_logit": bp0_logit.to(out.bp0.device),
        "bpI_logit": bpI_logit.to(out.bpI.device),
        "bp0_deriv": out.bp0.clamp(0.0, 1.0) * (1.0 - out.bp0.clamp(0.0, 1.0)),
        "bpI_deriv": out.bpI.clamp(0.0, 1.0) * (1.0 - out.bpI.clamp(0.0, 1.0)),
    }


def refit_loss(
    bundle: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    indices: torch.Tensor,
    hp,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    policy_delta = float(getattr(hp, "bp_grid_policy_huber_delta", 0.05))
    policy_weight = float(getattr(hp, "bp_grid_policy_weight", 1.0))
    mix_weight = float(getattr(hp, "bp_grid_mix_policy_weight", 1.0))
    parts = {}
    total = torch.zeros((), device=indices.device)
    for pred_key, target_key, branch_weight in [
        ("bp0", "bp0", policy_weight),
        ("bpI", "bpI", policy_weight),
        ("bp_mix", "bp_mix", mix_weight),
    ]:
        sample_weight = targets["bp_mix_survival_weight"] if target_key == "bp_mix" else None
        loss, _, _ = compute_target_grid_policy_distillation_loss(
            bundle[pred_key][indices],
            targets[target_key][indices],
            targets[f"{target_key}_confidence"][indices],
            huber_delta=policy_delta,
            branch_weight=branch_weight,
            sample_weight=sample_weight[indices] if sample_weight is not None else None,
        )
        parts[pred_key] = loss
        total = total + loss
    return total, parts


def eval_split(
    model: torch.nn.Module,
    parent_state: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    indices: torch.Tensor,
    hp,
) -> Dict[str, float]:
    if indices.numel() == 0:
        return {
            "loss": float("nan"),
            "mse": float("nan"),
            "mae": float("nan"),
            "bp0_mean": float("nan"),
            "bpI_mean": float("nan"),
            "bp_mix_mean": float("nan"),
            "bp0_logit_mean": float("nan"),
            "bpI_logit_mean": float("nan"),
            "bp0_sigmoid_derivative_mean": float("nan"),
            "bpI_sigmoid_derivative_mean": float("nan"),
        }
    bundle = forward_policy_bundle(model, parent_state)
    loss, _ = refit_loss(bundle, targets, indices, hp)
    pred = torch.cat([bundle["bp0"][indices], bundle["bpI"][indices], bundle["bp_mix"][indices]], dim=0)
    target = torch.cat([targets["bp0"][indices], targets["bpI"][indices], targets["bp_mix"][indices]], dim=0)
    return {
        "loss": float(loss.detach().item()),
        "mse": float((pred - target).pow(2).mean().detach().item()),
        "mae": float((pred - target).abs().mean().detach().item()),
        "bp0_mean": float(bundle["bp0"][indices].detach().mean().item()),
        "bpI_mean": float(bundle["bpI"][indices].detach().mean().item()),
        "bp_mix_mean": float(bundle["bp_mix"][indices].detach().mean().item()),
        "bp0_logit_mean": float(bundle["bp0_logit"][indices].detach().mean().item()),
        "bpI_logit_mean": float(bundle["bpI_logit"][indices].detach().mean().item()),
        "bp0_sigmoid_derivative_mean": float(bundle["bp0_deriv"][indices].detach().mean().item()),
        "bpI_sigmoid_derivative_mean": float(bundle["bpI_deriv"][indices].detach().mean().item()),
    }


def main() -> None:
    args = parse_args()
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    device = torch.device(args.device)
    Config.DEVICE = device
    run_root = args.run_root.resolve()
    firm_pkl = require_file(args.firm_pkl, "firm pickle")
    summary_path = require_file(args.decomposition_summary, "decomposition summary CSV")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = run_root / "checkpoints"
    require_file(ckpt_dir / f"ep{args.episode}_policy_value.pt", "policy/value checkpoint")
    hp = build_hyperparams()
    hp.pv_eta_resample_enabled = False
    hp.max_firm_train_units = 0
    lr = float(args.learning_rate if args.learning_rate is not None else getattr(hp, "policy_lr", 1e-3))

    models = build_models(device=device, ckpt_dir=ckpt_dir, ckpt_prefix=f"ep{args.episode}", strict=True)
    online_model = models["policy_value"].eval()
    probe_model = copy.deepcopy(online_model).to(device).eval()
    before_state = {name: p.detach().cpu().clone() for name, p in online_model.named_parameters()}
    probe_before = {name: p.detach().cpu().clone() for name, p in probe_model.named_parameters()}

    trainable = configure_refit_trainable_parameters(probe_model)
    print("Trainable parameters:")
    for name in trainable:
        print(f"  {name}")

    tensors = make_episode_batches(firm_pkl, online_model, hp, device, args.batch_size, args.n_branches)
    summary_df = pd.read_csv(summary_path)
    selected = select_exported_states(summary_df, tensors)
    parent_state = selected["parent"].detach()
    source_index = selected["source_index"].detach()
    targets = fixed_targets(summary_df, device, parent_state.dtype)
    splits = split_indices(parent_state.shape[0], args.holdout_fraction, args.seed, device)
    optimizer = torch.optim.AdamW(
        [p for p in probe_model.parameters() if p.requires_grad],
        lr=lr,
    )

    record_steps = sorted({s for s in args.record_steps if 0 <= s <= args.steps})
    if args.steps not in record_steps:
        record_steps.append(args.steps)
    record_set = set(record_steps)

    history_rows: List[Dict[str, object]] = []

    def record(step: int, grad_norms: Dict[str, float] | None = None) -> None:
        grad_norms = grad_norms or {
            "bp0_head_grad_norm": 0.0,
            "bpI_head_grad_norm": 0.0,
            "policy_encoder_grad_norm": 0.0,
        }
        probe_model.eval()
        with torch.no_grad():
            for split, idx in splits.items():
                metrics = eval_split(probe_model, parent_state, targets, idx, hp)
                history_rows.append(
                    {
                        "episode": args.episode,
                        "step": step,
                        "split": split,
                        **metrics,
                        **grad_norms,
                        "learning_rate": lr,
                    }
                )

    record(0)
    for step in range(1, args.steps + 1):
        probe_model.train()
        optimizer.zero_grad(set_to_none=True)
        bundle = forward_policy_bundle(probe_model, parent_state)
        loss, _ = refit_loss(bundle, targets, splits["train"], hp)
        loss.backward()
        grad_norms = {
            "bp0_head_grad_norm": float(sum(
                p.grad.detach().pow(2).sum().item()
                for p in probe_model.bp0_head.parameters()
                if p.grad is not None
            ) ** 0.5),
            "bpI_head_grad_norm": float(sum(
                p.grad.detach().pow(2).sum().item()
                for p in probe_model.bpi_head.parameters()
                if p.grad is not None
            ) ** 0.5),
            "policy_encoder_grad_norm": float(sum(
                p.grad.detach().pow(2).sum().item()
                for p in probe_model.policy_encoder.parameters()
                if p.grad is not None
            ) ** 0.5),
        }
        optimizer.step()
        if step in record_set:
            record(step, grad_norms)

    assert_only_allowed_changed(probe_before, probe_model)
    for name, param in online_model.named_parameters():
        if not torch.equal(before_state[name].to(param.device), param.detach()):
            raise AssertionError(f"Original online model parameter changed: {name}")

    with torch.no_grad():
        initial_bundle = forward_policy_bundle(online_model, parent_state)
        final_bundle = forward_policy_bundle(probe_model, parent_state)
    rows = []
    split_name = ["train"] * parent_state.shape[0]
    for idx in splits["holdout"].detach().cpu().tolist():
        split_name[idx] = "holdout"
    for i in range(parent_state.shape[0]):
        rows.append(
            {
                "episode": args.episode,
                "source_index": int(source_index[i].item()),
                "split": split_name[i],
                "bp0_initial": float(initial_bundle["bp0"][i].item()),
                "bpI_initial": float(initial_bundle["bpI"][i].item()),
                "bp_mix_initial": float(initial_bundle["bp_mix"][i].item()),
                "bp0_target": float(targets["bp0"][i].item()),
                "bpI_target": float(targets["bpI"][i].item()),
                "bp_mix_target": float(targets["bp_mix"][i].item()),
                "bp0_final": float(final_bundle["bp0"][i].item()),
                "bpI_final": float(final_bundle["bpI"][i].item()),
                "bp_mix_final": float(final_bundle["bp_mix"][i].item()),
            }
        )

    history = pd.DataFrame(history_rows)
    states = pd.DataFrame(rows)
    initial_train_mae = float(history[(history.step == 0) & (history.split == "train")]["mae"].iloc[0])
    final_train_mae = float(history[(history.step == args.steps) & (history.split == "train")]["mae"].iloc[-1])
    initial_holdout = history[(history.step == 0) & (history.split == "holdout")]["mae"]
    final_holdout = history[(history.step == args.steps) & (history.split == "holdout")]["mae"]
    initial_holdout_mae = float(initial_holdout.iloc[0]) if len(initial_holdout) else float("nan")
    final_holdout_mae = float(final_holdout.iloc[-1]) if len(final_holdout) else float("nan")
    initial_pred_mean = float(states[["bp0_initial", "bpI_initial", "bp_mix_initial"]].mean().mean())
    summary = pd.DataFrame(
        [
            {
                "episode": args.episode,
                "steps": args.steps,
                "learning_rate": lr,
                "n_states": int(parent_state.shape[0]),
                "n_train": int(splits["train"].numel()),
                "n_holdout": int(splits["holdout"].numel()),
                "initial_train_mae": initial_train_mae,
                "final_train_mae": final_train_mae,
                "initial_holdout_mae": initial_holdout_mae,
                "final_holdout_mae": final_holdout_mae,
                "bp0_initial_mean": float(states["bp0_initial"].mean()),
                "bp0_final_mean": float(states["bp0_final"].mean()),
                "bp0_target_mean": float(states["bp0_target"].mean()),
                "bpI_initial_mean": float(states["bpI_initial"].mean()),
                "bpI_final_mean": float(states["bpI_final"].mean()),
                "bpI_target_mean": float(states["bpI_target"].mean()),
                "recovered_from_low_saturation": bool(
                    initial_pred_mean < 0.01
                    and np.isfinite(final_holdout_mae)
                    and final_holdout_mae < 0.05
                ),
                "optimization_failure_suspected": bool(final_train_mae >= initial_train_mae * 0.95),
                "schedule_or_target_drift_suspected": bool(final_train_mae < initial_train_mae * 0.5),
            }
        ]
    )

    history.to_csv(output_dir / f"ep{args.episode}_bp_fixed_teacher_refit_history.csv", index=False)
    states.to_csv(output_dir / f"ep{args.episode}_bp_fixed_teacher_refit_states.csv", index=False)
    summary.to_csv(output_dir / f"ep{args.episode}_bp_fixed_teacher_refit_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
