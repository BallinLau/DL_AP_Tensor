from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from experiments.export_bp_deep_diagnostics import (  # noqa: E402
    make_episode_batches,
    select_exported_states,
)
from experiments.run_bp_fixed_teacher_refit_probe import (  # noqa: E402
    assert_only_allowed_changed,
    configure_refit_trainable_parameters,
    fixed_targets,
    parse_record_steps,
    require_file,
    split_indices,
)
from experiments.run_utils import build_hyperparams, build_models  # noqa: E402
from training.bp_policy_loss import (  # noqa: E402
    compute_target_grid_policy_distillation_loss,
    huber_element,
)


TRAINABLE_PREFIXES = ("policy_encoder.", "bp0_head.", "bpi_head.")
PROBE_BASELINE = "baseline_output_loss"
PROBE_RECENTER = "bias_recenter_output_loss"
PROBE_LOGIT = "original_init_logit_loss"


@dataclass
class ProbeConfig:
    episode: int
    steps: int
    record_steps: List[int]
    learning_rate: float
    weight_decay: float
    holdout_fraction: float
    seed: int
    initial_bp_target: float
    logit_target_eps: float
    logit_huber_delta: float
    probes: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled BP recovery probes without modifying checkpoints."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--firm-pkl", type=Path, required=True)
    parser.add_argument("--decomposition-summary", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--initial-bp-target", type=float, default=0.5)
    parser.add_argument("--logit-target-eps", type=float, default=1e-4)
    parser.add_argument("--logit-huber-delta", type=float, default=1.0)
    parser.add_argument(
        "--probes",
        type=str,
        default="baseline_output_loss,bias_recenter_output_loss,original_init_logit_loss",
    )
    parser.add_argument(
        "--record-steps",
        type=parse_record_steps,
        default=parse_record_steps("0,1,2,5,10,20,50,100,200"),
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-branches", type=int, default=2)
    parser.add_argument("--n-states", type=int, default=0)
    return parser.parse_args()


def parse_probes(value: str) -> List[str]:
    probes = [item.strip() for item in value.split(",") if item.strip()]
    valid = {PROBE_BASELINE, PROBE_RECENTER, PROBE_LOGIT}
    unknown = sorted(set(probes) - valid)
    if unknown:
        raise ValueError(f"Unknown probe(s): {unknown}")
    return probes


def last_linear(module: nn.Module) -> nn.Linear:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            return child
    raise RuntimeError(f"No Linear layer found in {module.__class__.__name__}")


def tensor_l2_grad_norm(module: nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().pow(2).sum().item())
    return float(total ** 0.5)


def stable_logit(bp: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.logit(bp.clamp(float(eps), 1.0 - float(eps)))


def forward_policy_bundle(model: torch.nn.Module, parent_state: torch.Tensor) -> Dict[str, torch.Tensor]:
    base_state, i, _ = model._split_state(parent_state)
    h_pi = model.policy_encoder(base_state)
    bp0_logit = model.bp0_head.forward_logits(h_pi)
    bpI_logit = model.bpi_head.forward_logits(h_pi, i)
    bp0 = torch.sigmoid(bp0_logit)
    bpI = torch.sigmoid(bpI_logit)
    out = model(parent_state)
    mix_weight = out.bar_i_cond.clamp(0.0, 1.0)
    bp_mix = (1.0 - mix_weight) * bp0 + mix_weight * bpI
    return {
        "bp0": bp0,
        "bpI": bpI,
        "bp_mix": bp_mix,
        "bp0_logit": bp0_logit,
        "bpI_logit": bpI_logit,
        "bp0_deriv": bp0 * (1.0 - bp0),
        "bpI_deriv": bpI * (1.0 - bpI),
        "mix_weight": mix_weight,
    }


def output_space_loss(
    bundle: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    indices: torch.Tensor,
    hp,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    policy_delta = float(getattr(hp, "bp_grid_policy_huber_delta", 0.05))
    policy_weight = float(getattr(hp, "bp_grid_policy_weight", 1.0))
    mix_weight = float(getattr(hp, "bp_grid_mix_policy_weight", 1.0))
    parts: Dict[str, torch.Tensor] = {}
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


def logit_space_loss(
    bundle: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    indices: torch.Tensor,
    hp,
    *,
    target_eps: float,
    huber_delta: float,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    policy_weight = float(getattr(hp, "bp_grid_policy_weight", 1.0))
    parts: Dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=indices.device)
    for branch in ["bp0", "bpI"]:
        pred = bundle[f"{branch}_logit"][indices]
        target = stable_logit(targets[branch][indices], target_eps)
        elem = huber_element(pred, target, huber_delta)
        loss = policy_weight * (elem * targets[f"{branch}_confidence"][indices]).mean()
        parts[branch] = loss
        total = total + loss
    return total, parts


def probe_loss(
    probe: str,
    bundle: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    indices: torch.Tensor,
    hp,
    cfg: ProbeConfig,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if probe == PROBE_LOGIT:
        return logit_space_loss(
            bundle,
            targets,
            indices,
            hp,
            target_eps=cfg.logit_target_eps,
            huber_delta=cfg.logit_huber_delta,
        )
    return output_space_loss(bundle, targets, indices, hp)


def branch_weights(targets: Dict[str, torch.Tensor], branch: str) -> torch.Tensor:
    if branch == "bp_mix":
        return targets["bp_mix_confidence"] * targets["bp_mix_survival_weight"]
    return targets[f"{branch}_confidence"]


def eval_branch_metrics(
    *,
    probe: str,
    bundle: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    indices: torch.Tensor,
    branch: str,
    split: str,
    step: int,
    hp,
    cfg: ProbeConfig,
    grad_norms: Dict[str, float],
) -> Dict[str, object]:
    pred = bundle[branch][indices]
    target = targets[branch][indices]
    weights = branch_weights(targets, branch)[indices]
    abs_err = (pred - target).abs()
    mse = (pred - target).pow(2).mean()
    weighted_mae = (weights * abs_err).sum() / weights.sum().clamp_min(1e-12)
    training_loss = torch.zeros((), device=pred.device)
    if branch in {"bp0", "bpI"}:
        target_logit = stable_logit(target, cfg.logit_target_eps)
        logit_abs = (bundle[f"{branch}_logit"][indices] - target_logit).abs()
        sigmoid_deriv = bundle[f"{branch}_deriv"][indices]
        if probe == PROBE_LOGIT:
            training_loss = (
                huber_element(bundle[f"{branch}_logit"][indices], target_logit, cfg.logit_huber_delta)
                * weights
            ).mean()
    else:
        logit_abs = torch.zeros_like(abs_err)
        sigmoid_deriv = torch.zeros_like(abs_err)
    if probe != PROBE_LOGIT:
        loss, _, _ = compute_target_grid_policy_distillation_loss(
            pred,
            target,
            targets[f"{branch}_confidence"][indices],
            huber_delta=float(getattr(hp, "bp_grid_policy_huber_delta", 0.05)),
            branch_weight=1.0,
            sample_weight=targets["bp_mix_survival_weight"][indices] if branch == "bp_mix" else None,
        )
        training_loss = loss
    return {
        "episode": cfg.episode,
        "probe": probe,
        "step": step,
        "split": split,
        "branch": branch.replace("bpI", "pi").replace("bp0", "p0").replace("bp_mix", "mix"),
        "n_states": int(indices.numel()),
        "training_loss": float(training_loss.detach().item()),
        "output_mse": float(mse.detach().item()),
        "output_mae": float(abs_err.mean().detach().item()),
        "weighted_output_mae": float(weighted_mae.detach().item()),
        "bp_pred_mean": float(pred.detach().mean().item()),
        "bp_pred_min": float(pred.detach().min().item()),
        "bp_pred_max": float(pred.detach().max().item()),
        "bp_target_mean": float(target.detach().mean().item()),
        "logit_abs_error_mean": float(logit_abs.detach().mean().item()),
        "sigmoid_derivative_mean": float(sigmoid_deriv.detach().mean().item()),
        **grad_norms,
        "learning_rate": cfg.learning_rate,
    }


def eval_combined_metrics(
    *,
    probe: str,
    bundle: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    indices: torch.Tensor,
    split: str,
    step: int,
    hp,
    cfg: ProbeConfig,
    grad_norms: Dict[str, float],
) -> Dict[str, object]:
    pred = torch.cat([bundle["bp0"][indices], bundle["bpI"][indices], bundle["bp_mix"][indices]], dim=0)
    target = torch.cat([targets["bp0"][indices], targets["bpI"][indices], targets["bp_mix"][indices]], dim=0)
    weights = torch.cat(
        [
            targets["bp0_confidence"][indices],
            targets["bpI_confidence"][indices],
            targets["bp_mix_confidence"][indices] * targets["bp_mix_survival_weight"][indices],
        ],
        dim=0,
    )
    loss, _ = probe_loss(probe, bundle, targets, indices, hp, cfg)
    abs_err = (pred - target).abs()
    return {
        "episode": cfg.episode,
        "probe": probe,
        "step": step,
        "split": split,
        "branch": "combined",
        "n_states": int(indices.numel()),
        "training_loss": float(loss.detach().item()),
        "output_mse": float((pred - target).pow(2).mean().detach().item()),
        "output_mae": float(abs_err.mean().detach().item()),
        "weighted_output_mae": float((weights * abs_err).sum().div(weights.sum().clamp_min(1e-12)).detach().item()),
        "bp_pred_mean": float(pred.detach().mean().item()),
        "bp_pred_min": float(pred.detach().min().item()),
        "bp_pred_max": float(pred.detach().max().item()),
        "bp_target_mean": float(target.detach().mean().item()),
        "logit_abs_error_mean": float(
            torch.cat(
                [
                    (bundle["bp0_logit"][indices] - stable_logit(targets["bp0"][indices], cfg.logit_target_eps)).abs(),
                    (bundle["bpI_logit"][indices] - stable_logit(targets["bpI"][indices], cfg.logit_target_eps)).abs(),
                ],
                dim=0,
            )
            .detach()
            .mean()
            .item()
        ),
        "sigmoid_derivative_mean": float(
            torch.cat([bundle["bp0_deriv"][indices], bundle["bpI_deriv"][indices]], dim=0)
            .detach()
            .mean()
            .item()
        ),
        **grad_norms,
        "learning_rate": cfg.learning_rate,
    }


def compute_grad_norms(
    model: torch.nn.Module,
    probe: str,
    parent_state: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    train_idx: torch.Tensor,
    hp,
    cfg: ProbeConfig,
) -> Dict[str, float]:
    model.zero_grad(set_to_none=True)
    bundle = forward_policy_bundle(model, parent_state)
    loss, _ = probe_loss(probe, bundle, targets, train_idx, hp, cfg)
    loss.backward()
    norms = {
        "bp0_head_grad_norm": tensor_l2_grad_norm(model.bp0_head),
        "bpI_head_grad_norm": tensor_l2_grad_norm(model.bpi_head),
        "policy_encoder_grad_norm": tensor_l2_grad_norm(model.policy_encoder),
    }
    model.zero_grad(set_to_none=True)
    return norms


def recenter_last_bias(
    *,
    head: nn.Module,
    logits: torch.Tensor,
    indices: torch.Tensor,
    target_bp: float,
    eps: float,
) -> Dict[str, float]:
    target_logit = float(torch.logit(torch.tensor(float(target_bp)).clamp(eps, 1.0 - eps)).item())
    median_before = float(logits[indices].detach().median().item())
    shift = target_logit - median_before
    layer = last_linear(head)
    with torch.no_grad():
        layer.bias.add_(shift)
    return {
        "target_logit": target_logit,
        "median_before": median_before,
        "bias_shift": float(shift),
    }


def apply_bias_recenter(
    model: torch.nn.Module,
    parent_state: torch.Tensor,
    train_idx: torch.Tensor,
    cfg: ProbeConfig,
) -> Dict[str, float]:
    before = forward_policy_bundle(model, parent_state)
    p0_stats = recenter_last_bias(
        head=model.bp0_head,
        logits=before["bp0_logit"],
        indices=train_idx,
        target_bp=cfg.initial_bp_target,
        eps=cfg.logit_target_eps,
    )
    pi_stats = recenter_last_bias(
        head=model.bpi_head,
        logits=before["bpI_logit"],
        indices=train_idx,
        target_bp=cfg.initial_bp_target,
        eps=cfg.logit_target_eps,
    )
    after = forward_policy_bundle(model, parent_state)
    return {
        "bp0_bias_shift": p0_stats["bias_shift"],
        "bpI_bias_shift": pi_stats["bias_shift"],
        "bp0_train_logit_median_before_recenter": p0_stats["median_before"],
        "bpI_train_logit_median_before_recenter": pi_stats["median_before"],
        "bp0_train_logit_median_after_recenter": float(after["bp0_logit"][train_idx].detach().median().item()),
        "bpI_train_logit_median_after_recenter": float(after["bpI_logit"][train_idx].detach().median().item()),
        "bp0_train_bp_median_after_recenter": float(after["bp0"][train_idx].detach().median().item()),
        "bpI_train_bp_median_after_recenter": float(after["bpI"][train_idx].detach().median().item()),
    }


def validate_recenter_translation(
    before: Dict[str, torch.Tensor],
    after: Dict[str, torch.Tensor],
    stats: Dict[str, float],
    train_idx: torch.Tensor,
) -> None:
    for branch, shift_key in [("bp0", "bp0_bias_shift"), ("bpI", "bpI_bias_shift")]:
        diff = after[f"{branch}_logit"][train_idx] - before[f"{branch}_logit"][train_idx]
        expected = torch.full_like(diff, float(stats[shift_key]))
        torch.testing.assert_close(diff, expected, rtol=1e-5, atol=1e-5)


def subset_targets(targets: Dict[str, torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
    return {key: value[:n].clone() for key, value in targets.items()}


def assert_online_unchanged(before: Dict[str, torch.Tensor], model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if not torch.equal(before[name].to(param.device), param.detach()):
            raise AssertionError(f"Online checkpoint model changed during probe: {name}")


def assert_finite_frame(df: pd.DataFrame, label: str) -> None:
    numeric = df.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        bad = numeric.columns[~np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=0)].tolist()
        raise RuntimeError(f"{label} contains non-finite numeric columns: {bad}")


def state_rows_for_probe(
    *,
    probe: str,
    cfg: ProbeConfig,
    parent_state: torch.Tensor,
    source_index: torch.Tensor,
    splits: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    initial_bundle: Dict[str, torch.Tensor],
    final_bundle: Dict[str, torch.Tensor],
    recenter_stats: Dict[str, float],
) -> List[Dict[str, object]]:
    split_name = ["train"] * parent_state.shape[0]
    for idx in splits["holdout"].detach().cpu().tolist():
        split_name[idx] = "holdout"
    rows: List[Dict[str, object]] = []
    for i in range(parent_state.shape[0]):
        rows.append(
            {
                "episode": cfg.episode,
                "probe": probe,
                "source_index": int(source_index[i].item()),
                "state_pos": i,
                "split": split_name[i],
                "b": float(parent_state[i, 0].item()),
                "z": float(parent_state[i, 1].item()),
                "eta": float(parent_state[i, 2].item()),
                "i": float(parent_state[i, 3].item()),
                "x": float(parent_state[i, 4].item()),
                "hatcf": float(parent_state[i, 5].item()),
                "lnkf": float(parent_state[i, 6].item()),
                "bp0_initial": float(initial_bundle["bp0"][i].detach().item()),
                "bpI_initial": float(initial_bundle["bpI"][i].detach().item()),
                "bp_mix_initial": float(initial_bundle["bp_mix"][i].detach().item()),
                "bp0_final": float(final_bundle["bp0"][i].detach().item()),
                "bpI_final": float(final_bundle["bpI"][i].detach().item()),
                "bp_mix_final": float(final_bundle["bp_mix"][i].detach().item()),
                "bp0_target": float(targets["bp0"][i].item()),
                "bpI_target": float(targets["bpI"][i].item()),
                "bp_mix_target": float(targets["bp_mix"][i].item()),
                "bp0_initial_logit": float(initial_bundle["bp0_logit"][i].detach().item()),
                "bpI_initial_logit": float(initial_bundle["bpI_logit"][i].detach().item()),
                "bp0_final_logit": float(final_bundle["bp0_logit"][i].detach().item()),
                "bpI_final_logit": float(final_bundle["bpI_logit"][i].detach().item()),
                "bp0_target_logit": float(stable_logit(targets["bp0"][i], cfg.logit_target_eps).item()),
                "bpI_target_logit": float(stable_logit(targets["bpI"][i], cfg.logit_target_eps).item()),
                "bp0_confidence": float(targets["bp0_confidence"][i].item()),
                "bpI_confidence": float(targets["bpI_confidence"][i].item()),
                "bp_mix_confidence": float(targets["bp_mix_confidence"][i].item()),
                "bp_mix_survival_weight": float(targets["bp_mix_survival_weight"][i].item()),
                "mix_weight_initial": float(initial_bundle["mix_weight"][i].detach().item()),
                "mix_weight_final": float(final_bundle["mix_weight"][i].detach().item()),
                "bp0_bias_shift": float(recenter_stats.get("bp0_bias_shift", 0.0)),
                "bpI_bias_shift": float(recenter_stats.get("bpI_bias_shift", 0.0)),
                "bp0_train_logit_median_after_recenter": float(
                    recenter_stats.get("bp0_train_logit_median_after_recenter", 0.0)
                ),
                "bpI_train_logit_median_after_recenter": float(
                    recenter_stats.get("bpI_train_logit_median_after_recenter", 0.0)
                ),
            }
        )
    return rows


def make_summary_rows(history: pd.DataFrame, states: pd.DataFrame, cfg: ProbeConfig) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for probe in sorted(history["probe"].unique()):
        h = history[(history["probe"] == probe) & (history["branch"] == "combined")]
        s = states[states["probe"] == probe]
        initial_train = h[(h["step"] == 0) & (h["split"] == "train")].iloc[0]
        final_train = h[(h["step"] == cfg.steps) & (h["split"] == "train")].iloc[-1]
        initial_holdout_rows = h[(h["step"] == 0) & (h["split"] == "holdout")]
        final_holdout_rows = h[(h["step"] == cfg.steps) & (h["split"] == "holdout")]
        initial_holdout = initial_holdout_rows.iloc[0] if len(initial_holdout_rows) else initial_train
        final_holdout = final_holdout_rows.iloc[-1] if len(final_holdout_rows) else final_train
        initial_train_mae = float(initial_train["output_mae"])
        final_train_mae = float(final_train["output_mae"])
        initial_holdout_mae = float(initial_holdout["output_mae"])
        final_holdout_mae = float(final_holdout["output_mae"])
        rows.append(
            {
                "episode": cfg.episode,
                "probe": probe,
                "steps": cfg.steps,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "n_states": int(len(s)),
                "n_train": int((s["split"] == "train").sum()),
                "n_holdout": int((s["split"] == "holdout").sum()),
                "initial_train_mae": initial_train_mae,
                "final_train_mae": final_train_mae,
                "train_mae_improvement": initial_train_mae - final_train_mae,
                "train_mae_improvement_ratio": (initial_train_mae - final_train_mae) / max(initial_train_mae, 1e-12),
                "initial_holdout_mae": initial_holdout_mae,
                "final_holdout_mae": final_holdout_mae,
                "holdout_mae_improvement": initial_holdout_mae - final_holdout_mae,
                "holdout_mae_improvement_ratio": (initial_holdout_mae - final_holdout_mae)
                / max(initial_holdout_mae, 1e-12),
                "initial_weighted_train_mae": float(initial_train["weighted_output_mae"]),
                "final_weighted_train_mae": float(final_train["weighted_output_mae"]),
                "initial_weighted_holdout_mae": float(initial_holdout["weighted_output_mae"]),
                "final_weighted_holdout_mae": float(final_holdout["weighted_output_mae"]),
                "initial_logit_abs_error": float(initial_train["logit_abs_error_mean"]),
                "final_logit_abs_error": float(final_train["logit_abs_error_mean"]),
                "initial_bp_head_grad_norm": float(
                    initial_train["bp0_head_grad_norm"] + initial_train["bpI_head_grad_norm"]
                ),
                "final_bp_head_grad_norm": float(final_train["bp0_head_grad_norm"] + final_train["bpI_head_grad_norm"]),
                "bp0_initial_mean": float(s["bp0_initial"].mean()),
                "bp0_final_mean": float(s["bp0_final"].mean()),
                "bp0_target_mean": float(s["bp0_target"].mean()),
                "bpI_initial_mean": float(s["bpI_initial"].mean()),
                "bpI_final_mean": float(s["bpI_final"].mean()),
                "bpI_target_mean": float(s["bpI_target"].mean()),
                "bp_mix_initial_mean": float(s["bp_mix_initial"].mean()),
                "bp_mix_final_mean": float(s["bp_mix_final"].mean()),
                "bp_mix_target_mean": float(s["bp_mix_target"].mean()),
                "recovered_from_low_saturation": bool(
                    s[["bp0_initial", "bpI_initial"]].to_numpy().mean() < 0.01
                    and final_train_mae < max(0.05, initial_train_mae * 0.5)
                ),
                "optimization_failure_suspected": bool(final_train_mae >= initial_train_mae * 0.95),
                "schedule_or_target_drift_suspected": bool(final_train_mae < initial_train_mae * 0.5),
            }
        )
    return pd.DataFrame(rows)


def run_probe_suite(
    *,
    online_model: torch.nn.Module,
    parent_state: torch.Tensor,
    source_index: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    hp,
    cfg: ProbeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    torch.manual_seed(cfg.seed)
    online_before = {name: param.detach().cpu().clone() for name, param in online_model.named_parameters()}
    target_before = {name: value.detach().clone() for name, value in targets.items()}
    splits = split_indices(parent_state.shape[0], cfg.holdout_fraction, cfg.seed, parent_state.device)
    history_rows: List[Dict[str, object]] = []
    state_rows: List[Dict[str, object]] = []

    for probe in cfg.probes:
        torch.manual_seed(cfg.seed)
        probe_model = copy.deepcopy(online_model).to(parent_state.device)
        configure_refit_trainable_parameters(probe_model)
        probe_initial_params = {name: param.detach().cpu().clone() for name, param in probe_model.named_parameters()}
        recenter_stats: Dict[str, float] = {}
        initial_bundle = forward_policy_bundle(probe_model, parent_state)
        if probe == PROBE_RECENTER:
            before_recenter = {key: value.detach().clone() for key, value in initial_bundle.items()}
            recenter_stats = apply_bias_recenter(probe_model, parent_state, splits["train"], cfg)
            initial_bundle = forward_policy_bundle(probe_model, parent_state)
            validate_recenter_translation(before_recenter, initial_bundle, recenter_stats, splits["train"])

        optimizer = torch.optim.AdamW(
            [param for param in probe_model.parameters() if param.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        record_steps = sorted({step for step in cfg.record_steps if 0 <= step <= cfg.steps})
        if cfg.steps not in record_steps:
            record_steps.append(cfg.steps)
        record_set = set(record_steps)

        def record(step: int, grad_norms: Dict[str, float]) -> None:
            probe_model.eval()
            with torch.no_grad():
                bundle = forward_policy_bundle(probe_model, parent_state)
                for split, idx in splits.items():
                    if idx.numel() == 0:
                        continue
                    for branch in ["bp0", "bpI", "bp_mix"]:
                        history_rows.append(
                            eval_branch_metrics(
                                probe=probe,
                                bundle=bundle,
                                targets=targets,
                                indices=idx,
                                branch=branch,
                                split=split,
                                step=step,
                                hp=hp,
                                cfg=cfg,
                                grad_norms=grad_norms,
                            )
                        )
                    history_rows.append(
                        eval_combined_metrics(
                            probe=probe,
                            bundle=bundle,
                            targets=targets,
                            indices=idx,
                            split=split,
                            step=step,
                            hp=hp,
                            cfg=cfg,
                            grad_norms=grad_norms,
                        )
                    )

        grad0 = compute_grad_norms(probe_model, probe, parent_state, targets, splits["train"], hp, cfg)
        record(0, grad0)
        last_grad = grad0
        for step in range(1, cfg.steps + 1):
            probe_model.train()
            optimizer.zero_grad(set_to_none=True)
            bundle = forward_policy_bundle(probe_model, parent_state)
            loss, _ = probe_loss(probe, bundle, targets, splits["train"], hp, cfg)
            loss.backward()
            last_grad = {
                "bp0_head_grad_norm": tensor_l2_grad_norm(probe_model.bp0_head),
                "bpI_head_grad_norm": tensor_l2_grad_norm(probe_model.bpi_head),
                "policy_encoder_grad_norm": tensor_l2_grad_norm(probe_model.policy_encoder),
            }
            optimizer.step()
            if step in record_set:
                record(step, last_grad)

        assert_only_allowed_changed(probe_initial_params, probe_model)
        assert_online_unchanged(online_before, online_model)
        for key, value in targets.items():
            torch.testing.assert_close(value, target_before[key], rtol=0, atol=0)

        final_bundle = forward_policy_bundle(probe_model, parent_state)
        state_rows.extend(
            state_rows_for_probe(
                probe=probe,
                cfg=cfg,
                parent_state=parent_state,
                source_index=source_index,
                splits=splits,
                targets=targets,
                initial_bundle=initial_bundle,
                final_bundle=final_bundle,
                recenter_stats=recenter_stats,
            )
        )

    history = pd.DataFrame(history_rows)
    states = pd.DataFrame(state_rows)
    summary = make_summary_rows(history, states, cfg)
    assert_finite_frame(history, "history")
    assert_finite_frame(states, "states")
    assert_finite_frame(summary, "summary")
    return history, states, summary


def main() -> None:
    args = parse_args()
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if not (0.0 < args.initial_bp_target < 1.0):
        raise ValueError("--initial-bp-target must be in (0, 1)")
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
    weight_decay = float(args.weight_decay if args.weight_decay is not None else getattr(hp, "policy_weight_decay", 0.0))
    cfg = ProbeConfig(
        episode=args.episode,
        steps=args.steps,
        record_steps=args.record_steps,
        learning_rate=lr,
        weight_decay=weight_decay,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        initial_bp_target=args.initial_bp_target,
        logit_target_eps=args.logit_target_eps,
        logit_huber_delta=args.logit_huber_delta,
        probes=parse_probes(args.probes),
    )

    models = build_models(device=device, ckpt_dir=ckpt_dir, ckpt_prefix=f"ep{args.episode}", strict=True)
    online_model = models["policy_value"].eval()
    tensors = make_episode_batches(firm_pkl, online_model, hp, device, args.batch_size, args.n_branches)
    summary_df = pd.read_csv(summary_path)
    selected = select_exported_states(summary_df, tensors)
    parent_state = selected["parent"].detach()
    source_index = selected["source_index"].detach()
    targets = fixed_targets(summary_df, device, parent_state.dtype)
    if args.n_states > 0:
        n_states = min(args.n_states, parent_state.shape[0])
        parent_state = parent_state[:n_states]
        source_index = source_index[:n_states]
        targets = subset_targets(targets, n_states)

    history, states, summary = run_probe_suite(
        online_model=online_model,
        parent_state=parent_state,
        source_index=source_index,
        targets=targets,
        hp=hp,
        cfg=cfg,
    )

    history_path = output_dir / f"ep{args.episode}_bp_recovery_probe_history.csv"
    states_path = output_dir / f"ep{args.episode}_bp_recovery_probe_states.csv"
    summary_path_out = output_dir / f"ep{args.episode}_bp_recovery_probe_summary.csv"
    history.to_csv(history_path, index=False)
    states.to_csv(states_path, index=False)
    summary.to_csv(summary_path_out, index=False)
    print(f"Wrote {history_path}")
    print(f"Wrote {states_path}")
    print(f"Wrote {summary_path_out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
