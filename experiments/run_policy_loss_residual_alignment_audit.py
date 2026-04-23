"""
Audit whether policy/value training losses align with Bellman residual metrics.

This script intentionally avoids outer episodes and FC2. It builds one fixed
policy tuple support, trains policy_value on that support, and periodically
records both:

- train-objective quantities, such as AIO main losses and total loss terms
- raw Bellman residual quantities used by evaluate_bellman_convergence

The goal is to test whether "loss goes down" means "Bellman residual goes down"
under the current P0 / PI / Q loss definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from data import SimulateTS  # noqa: E402
from experiments.run_policy_datagen_compare import (  # noqa: E402
    _init_parent_pool,
    _policy_checkpoint_status,
    _set_seed,
    generate_policy_tuple_tables,
)
from experiments.run_utils import build_hyperparams, build_models, build_optimizers  # noqa: E402
from losses.utils import compute_aio_residual  # noqa: E402
from training.episode import Episode  # noqa: E402


def _to_float(value: Any) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().mean().item())
    try:
        return float(value)
    except Exception:
        return float("nan")


def _tensor_stats(prefix: str, values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().reshape(-1).to(torch.float32)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {
            f"{prefix}_n": 0.0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_n": float(values.numel()),
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_p50": float(torch.quantile(values, 0.50).item()),
        f"{prefix}_p90": float(torch.quantile(values, 0.90).item()),
        f"{prefix}_max": float(values.max().item()),
    }


def _flatten_abs(residuals: Sequence[torch.Tensor]) -> torch.Tensor:
    chunks = [r.detach().abs().reshape(-1) for r in residuals if r is not None and r.numel() > 0]
    if not chunks:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def _summarize_residual_family(
    prefix: str,
    residuals: Sequence[torch.Tensor],
    aio_weight: float,
) -> Dict[str, float]:
    out = _tensor_stats(f"{prefix}_raw_abs", _flatten_abs(residuals))
    aio = compute_aio_residual(list(residuals), aio_weight).detach().reshape(-1)
    out.update(_tensor_stats(f"{prefix}_aio", aio))
    return out


def _get_m_lists(
    episode: Episode,
    parent: torch.Tensor,
    children: Sequence[torch.Tensor],
    kind: str,
    use_train_m: bool,
) -> List[torch.Tensor]:
    if parent.shape[1] > 7:
        raw_m = [child[:, 7:8] for child in children]
    else:
        raw_m = [torch.ones(parent.shape[0], 1, device=episode.device) for _ in children]

    if not use_train_m:
        return raw_m

    hp = episode.hyperparams
    if kind in {"p0", "pi"}:
        if bool(getattr(hp, "pv_use_clipped_m", True)):
            lo = float(getattr(hp, "pv_m_clamp_min", 0.7))
            hi = float(getattr(hp, "pv_m_clamp_max", 1.3))
            return [m.clamp(lo, hi) for m in raw_m]
        return raw_m

    if kind == "q":
        if bool(getattr(hp, "q_use_detached_m", True)):
            lo = float(getattr(hp, "q_m_clamp_min", 0.5))
            hi = float(getattr(hp, "q_m_clamp_max", 1.5))
            return [m.clamp(lo, hi) for m in raw_m]
        return raw_m

    raise ValueError(f"Unknown residual kind: {kind}")


def _p0_pi_residuals(
    episode: Episode,
    batch: Dict[str, torch.Tensor],
    kind: str,
    use_train_m: bool,
) -> List[torch.Tensor]:
    if kind not in {"p0", "pi"}:
        raise ValueError(kind)

    model = episode.models["policy_value"]
    loss_fn = episode.loss_fns[kind]
    parent = batch["parent"]
    children = episode._get_policy_children(batch)
    if not children:
        return []

    parent_state = episode._policy_strip_extra(parent)
    output_t = model(parent_state)
    bp0_t = episode._policy_get_out(output_t, "bp0", 1)
    bpI_t = episode._policy_get_out(output_t, "bpI", 2)
    bp_use = bp0_t if kind == "p0" else bpI_t

    output_children = []
    eta_children = []
    for child in children:
        child_state = episode._policy_strip_extra(child).clone()
        eta_child = child[:, 2:3]
        child_state[:, 0:1] = bp_use
        output_children.append(model(child_state))
        eta_children.append(eta_child)

    child_policy_state = parent_state.clone()
    child_policy_state[:, 0:1] = bp_use
    output_policy_child = model(child_policy_state)

    q_parent = episode._policy_get_out(output_t, "Q", 0)
    q_policy_child = episode._policy_get_out(output_policy_child, "Q", 0)
    p_children = [episode._policy_get_out(out, "P", 7) for out in output_children]
    bar_z_children = [episode._policy_get_out(out, "bar_z", 6) for out in output_children]
    m_list = _get_m_lists(episode, parent, children, kind=kind, use_train_m=use_train_m)

    if kind == "p0":
        value = episode._policy_get_out(output_t, "P0", 3)
        cashflows = [
            loss_fn.compute_cashflow_p0(
                parent_state[:, 4:5],
                parent_state[:, 1:2],
                parent_state[:, 0:1],
                q_parent,
                q_policy_child,
                eta_j,
            )
            for eta_j in eta_children
        ]
    else:
        value = episode._policy_get_out(output_t, "PI", 4)
        cashflows = [
            loss_fn.compute_cashflow_pi(
                parent_state[:, 4:5],
                parent_state[:, 1:2],
                parent_state[:, 0:1],
                parent_state[:, 3:4],
                q_parent,
                q_policy_child,
                eta_j,
            )
            for eta_j in eta_children
        ]

    return loss_fn.compute_bellman_residual(value, cashflows, m_list, p_children, bar_z_children)


def _q_residuals(
    episode: Episode,
    batch: Dict[str, torch.Tensor],
    use_train_m: bool,
) -> List[torch.Tensor]:
    model = episode.models["policy_value"]
    loss_fn = episode.loss_fns["q"]
    parent = batch["parent"]
    children = episode._get_policy_children(batch)
    if not children:
        return []

    parent_state = episode._policy_strip_extra(parent)
    output_t = model(parent_state)
    bp0_t = episode._policy_get_out(output_t, "bp0", 1)
    bpI_t = episode._policy_get_out(output_t, "bpI", 2)
    bar_i_t = episode._policy_get_out(output_t, "bar_i", 5)
    bp_t = episode._policy_get_out(output_t, "bp", -1)
    if bp_t.shape != bp0_t.shape:
        bp_t = bar_i_t * bpI_t + (1 - bar_i_t) * bp0_t

    b_parent = parent_state[:, 0:1]
    multiplier = bar_i_t * (float(getattr(loss_fn, "g", 1.0)) - 1.0) + 1.0
    b_sp = b_parent / multiplier.clamp_min(1e-6)

    outputsp_children = []
    for child in children:
        child_state = episode._policy_strip_extra(child).clone()
        child_state[:, 0:1] = b_sp
        outputsp_children.append(model(child_state))

    q_parent = episode._policy_get_out(output_t, "Q", 0)
    qsp_children = [episode._policy_get_out(out, "Q", 0) for out in outputsp_children]
    bar_zsp_children = [episode._policy_get_out(out, "bar_z", 6) for out in outputsp_children]
    x_children = [child[:, 4:5] for child in children]
    z_children = [child[:, 1:2] for child in children]
    m_list = _get_m_lists(episode, parent, children, kind="q", use_train_m=use_train_m)

    return loss_fn.compute_main_residual(
        q_parent,
        b_parent,
        bar_i_t,
        m_list,
        qsp_children,
        bar_zsp_children,
        x_children,
        z_children,
    )


def _aggregate_dicts(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row.keys()})
    out: Dict[str, float] = {}
    for key in keys:
        vals = np.array([row[key] for row in rows if key in row and np.isfinite(row[key])], dtype=float)
        if vals.size:
            out[key] = float(vals.mean())
    return out


def _compute_loss_terms_on_batch(
    episode: Episode,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    model = episode.models["policy_value"]
    was_training = model.training
    model.eval()

    with torch.enable_grad():
        p0_loss = episode._compute_p0_loss(batch)
        out["eval_p0_total_loss"] = _to_float(p0_loss)
        out.update({f"eval_{k}": _to_float(v) for k, v in episode._latest_p0_terms.items()})

        pi_loss = episode._compute_pi_loss(batch)
        out["eval_pi_total_loss"] = _to_float(pi_loss)
        out.update({f"eval_{k}": _to_float(v) for k, v in episode._latest_pi_terms.items()})

        q_loss = episode._compute_q_loss(batch)
        out["eval_q_total_loss"] = _to_float(q_loss)
        out.update({f"eval_{k}": _to_float(v) for k, v in episode._latest_q_terms.items()})

    for optimizer in episode.optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return out


def _compute_alignment_metrics(
    episode: Episode,
    metric_batches: Sequence[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    model = episode.models["policy_value"]
    was_training = model.training
    model.eval()

    for batch in metric_batches:
        row: Dict[str, float] = {}
        with torch.no_grad():
            for kind in ("p0", "pi"):
                train_res = _p0_pi_residuals(episode, batch, kind=kind, use_train_m=True)
                raw_res = _p0_pi_residuals(episode, batch, kind=kind, use_train_m=False)
                if train_res:
                    row.update(
                        _summarize_residual_family(
                            f"{kind}_trainM",
                            train_res,
                            episode.loss_fns[kind].aio_weight,
                        )
                    )
                if raw_res:
                    row.update(
                        _summarize_residual_family(
                            f"{kind}_rawM",
                            raw_res,
                            episode.loss_fns[kind].aio_weight,
                        )
                    )

            q_train_res = _q_residuals(episode, batch, use_train_m=True)
            q_raw_res = _q_residuals(episode, batch, use_train_m=False)
            if q_train_res:
                row.update(
                    _summarize_residual_family(
                        "q_trainM",
                        q_train_res,
                        episode.loss_fns["q"].aio_weight,
                    )
                )
            if q_raw_res:
                row.update(
                    _summarize_residual_family(
                        "q_rawM",
                        q_raw_res,
                        episode.loss_fns["q"].aio_weight,
                    )
                )

        row.update(_compute_loss_terms_on_batch(episode, batch))
        rows.append(row)

    if was_training:
        model.train()
    return _aggregate_dicts(rows)


def _extract_conv(convergence: Dict[str, Any]) -> Dict[str, float]:
    eq = convergence.get("equations", {})
    out: Dict[str, float] = {}
    for name in ("p0", "pi", "q"):
        item = eq.get(name, {})
        out[f"conv_{name}_mean"] = float(item.get("mean", float("nan")))
        out[f"conv_{name}_p90"] = float(item.get("p90", float("nan")))
    return out


def _average_losses(loss_rows: Sequence[Dict[str, float]], prefix: str) -> Dict[str, float]:
    if not loss_rows:
        return {}
    out = _aggregate_dicts(loss_rows)
    return {f"{prefix}_{k}": v for k, v in out.items()}


def _record_snapshot(
    episode: Episode,
    metric_batches: Sequence[Dict[str, torch.Tensor]],
    history: List[Dict[str, float]],
    phase: str,
    epoch: int,
    recent_losses: Sequence[Dict[str, float]],
) -> None:
    conv = episode.evaluate_bellman_convergence(list(metric_batches))
    row: Dict[str, float] = {
        "epoch": float(epoch),
        "phase_id": float({"initial": 0, "q_stage": 1, "pvbp_stage": 2, "q_refresh": 3}.get(phase, -1)),
        "phase": phase,
    }
    row.update(_extract_conv(conv))
    row.update(_compute_alignment_metrics(episode, metric_batches))
    row.update(_average_losses(recent_losses, "recent_train"))
    history.append(row)


def _set_stage_flags(
    episode: Episode,
    phase: str,
    epoch: int,
    q_stage_epochs: int,
) -> List[str]:
    episode._current_epoch_idx = int(epoch)
    episode._q_only_stage = phase == "q_stage"
    episode._pvbp_only_stage = phase == "pvbp_stage"
    episode._q_refresh_stage = phase == "q_refresh"
    episode._bp_only_stage = False
    episode._value_only_stage = False
    episode._set_policy_runtime_controls(q_stage_epochs)
    if phase in {"q_stage", "q_refresh"}:
        return ["q"]
    if phase == "pvbp_stage":
        return ["p0", "pi"]
    raise ValueError(f"Unknown phase: {phase}")


def _run_epoch_pass(
    episode: Episode,
    batches: Sequence[Dict[str, torch.Tensor]],
    phase: str,
    policy_terms: List[str],
    value_only: bool = False,
    bp_only: bool = False,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    prev_value_only = episode._value_only_stage
    prev_bp_only = episode._bp_only_stage
    episode._value_only_stage = value_only
    episode._bp_only_stage = bp_only
    try:
        for batch in tqdm(batches, desc=f"{phase}", leave=False):
            rows.append(
                episode.train_step(
                    batch,
                    train_modules=["policy_value"],
                    policy_loss_terms=policy_terms,
                )
            )
    finally:
        episode._value_only_stage = prev_value_only
        episode._bp_only_stage = prev_bp_only
    return rows


def _plot_history(history: Sequence[Dict[str, float]], out_dir: Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), sharex=True)
    specs = [
        ("p0", "eval_p0_main", "p0_rawM_raw_abs_mean"),
        ("pi", "eval_pi_main", "pi_rawM_raw_abs_mean"),
        ("q", "eval_q_main", "q_rawM_raw_abs_mean"),
    ]
    for ax, (name, loss_key, raw_key) in zip(axes, specs):
        ax.plot(epochs, [row.get(loss_key, np.nan) for row in history], marker="o", label=loss_key)
        ax.plot(epochs, [row.get(raw_key, np.nan) for row in history], marker="s", label=raw_key)
        ax.plot(epochs, [row.get(f"conv_{name}_mean", np.nan) for row in history], marker="^", label=f"conv_{name}_mean")
        ax.set_title(name.upper())
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("metric value")
    axes[1].set_xlabel("epoch")
    fig.suptitle("Policy Loss vs Raw Bellman Residual Alignment")
    fig.tight_layout()
    fig.savefig(out_dir / "alignment_loss_vs_residual.png", dpi=150)
    plt.close(fig)

    plt.figure(figsize=(7.5, 4.5))
    for name in ("p0", "pi", "q"):
        plt.plot(epochs, [row.get(f"conv_{name}_mean", np.nan) for row in history], marker="o", label=f"{name} conv mean")
    plt.xlabel("epoch")
    plt.ylabel("raw abs Bellman residual mean")
    plt.title("Convergence Residuals on Fixed Audit Batches")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "alignment_conv_means.png", dpi=150)
    plt.close()


def _write_history_csv(history: Sequence[Dict[str, float]], path: Path) -> None:
    if not history:
        return
    keys = sorted({k for row in history for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _phase_for_epoch(epoch: int, q_stage_epochs: int, pvbp_stage_epochs: int) -> str:
    if epoch < q_stage_epochs:
        return "q_stage"
    if epoch < q_stage_epochs + pvbp_stage_epochs:
        return "pvbp_stage"
    return "q_refresh"


def _require_policy_checkpoint(status: Dict[str, Any], allow_random: bool) -> None:
    if status.get("policy_value_loaded", False) or allow_random:
        return
    raise RuntimeError(
        "Policy checkpoint is required but was not found. "
        f"status={status}. Pass --allow-random-policy-init only for smoke tests."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit policy loss/residual alignment on fixed Bellman tuples.")
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--ckpt-prefix", type=str, default=None)
    parser.add_argument("--allow-random-policy-init", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--datagen-mode", choices=["main_branch", "children_promote"], default="main_branch")
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--eval-pool-size", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--eval-rollout-steps", type=int, default=None)
    parser.add_argument("--branch-num", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--metric-batch-cap", type=int, default=4)
    parser.add_argument("--q-stage-epochs", type=int, default=100)
    parser.add_argument("--pvbp-stage-epochs", type=int, default=100)
    parser.add_argument("--q-refresh-stage-epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = ROOT / "experiments" / "policy_loss_residual_alignment_audit"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    _set_seed(args.seed)

    ckpt_status = _policy_checkpoint_status(args.ckpt_dir, args.ckpt_prefix)
    _require_policy_checkpoint(ckpt_status, allow_random=args.allow_random_policy_init)

    models = build_models(device=device, ckpt_dir=args.ckpt_dir, ckpt_prefix=args.ckpt_prefix)
    hp = build_hyperparams()
    hp.q_stage_epochs = int(args.q_stage_epochs)
    hp.pvbp_stage_epochs = int(args.pvbp_stage_epochs)
    hp.q_refresh_stage_epochs = int(args.q_refresh_stage_epochs)
    hp.q_pretrain_epochs = int(args.q_stage_epochs)
    hp.q_warmstart_epochs = int(args.q_stage_epochs)
    hp.epochs = int(args.q_stage_epochs + args.pvbp_stage_epochs + args.q_refresh_stage_epochs)
    optimizers = build_optimizers(models, hp)
    episode = Episode(
        models=models,
        optimizers=optimizers,
        config=Config,
        hyperparams=hp,
        device=device,
        episode_id=0,
    )

    sim = SimulateTS(
        models=models,
        config=Config,
        n_paths=args.pool_size,
        group_size=args.group_size,
        horizon=1,
        branch_num=args.branch_num,
        main_branch=0,
        enable_entry=True,
        enable_exit=True,
        fc2_as_main_macro_state=False,
        device=device,
    )
    train_pool = _init_parent_pool(sim, pool_size=args.pool_size, seed=args.seed)
    train_result = generate_policy_tuple_tables(
        sim=sim,
        initial_parent_pool=train_pool,
        rollout_steps=args.rollout_steps,
        mode=args.datagen_mode,
        seed=args.seed + 1,
    )
    train_batches = episode._create_firm_batches_from_tensor(
        train_result.firm_table,
        batch_size=args.batch_size,
        n_branches=args.branch_num,
        eta_resample=True,
    )
    if not train_batches:
        raise RuntimeError("No training batches created for alignment audit.")

    eval_pool_size = int(args.pool_size if args.eval_pool_size is None else args.eval_pool_size)
    eval_rollout_steps = int(args.rollout_steps if args.eval_rollout_steps is None else args.eval_rollout_steps)
    eval_pool = _init_parent_pool(sim, pool_size=eval_pool_size, seed=args.seed + 9000)
    eval_result = generate_policy_tuple_tables(
        sim=sim,
        initial_parent_pool=eval_pool,
        rollout_steps=eval_rollout_steps,
        mode=args.datagen_mode,
        seed=args.seed + 9001,
    )
    eval_batches = episode._create_firm_batches_from_tensor(
        eval_result.firm_table,
        batch_size=args.batch_size,
        n_branches=args.branch_num,
        eta_resample=False,
    )
    if not eval_batches:
        raise RuntimeError("No eval batches created for alignment audit.")
    metric_batches = eval_batches[: max(1, int(args.metric_batch_cap))]

    total_epochs = int(args.q_stage_epochs + args.pvbp_stage_epochs + args.q_refresh_stage_epochs)
    history: List[Dict[str, float]] = []
    _record_snapshot(
        episode=episode,
        metric_batches=metric_batches,
        history=history,
        phase="initial",
        epoch=0,
        recent_losses=[],
    )

    recent_losses: List[Dict[str, float]] = []
    for epoch in range(total_epochs):
        phase = _phase_for_epoch(epoch, int(args.q_stage_epochs), int(args.pvbp_stage_epochs))
        policy_terms = _set_stage_flags(episode, phase, epoch, int(args.q_stage_epochs))

        if phase == "pvbp_stage" and bool(getattr(hp, "pvbp_alternating_enabled", True)):
            epoch_losses = []
            epoch_losses.extend(
                _run_epoch_pass(episode, train_batches, f"{phase}:value", policy_terms, value_only=True)
            )
            epoch_losses.extend(
                _run_epoch_pass(episode, train_batches, f"{phase}:bp", policy_terms, bp_only=True)
            )
        else:
            epoch_losses = _run_epoch_pass(episode, train_batches, phase, policy_terms)

        recent_losses.extend(epoch_losses)
        is_stage_end = (
            epoch + 1 == int(args.q_stage_epochs)
            or epoch + 1 == int(args.q_stage_epochs + args.pvbp_stage_epochs)
            or epoch + 1 == total_epochs
        )
        if (epoch + 1) % max(1, int(args.eval_every)) == 0 or is_stage_end:
            _record_snapshot(
                episode=episode,
                metric_batches=metric_batches,
                history=history,
                phase=phase,
                epoch=epoch + 1,
                recent_losses=recent_losses,
            )
            recent_losses = []

    episode._q_only_stage = False
    episode._q_refresh_stage = False
    episode._pvbp_only_stage = False
    episode._bp_only_stage = False
    episode._value_only_stage = False
    episode._set_policy_runtime_controls(0)

    manifest = {
        "ckpt_dir": str(args.ckpt_dir) if args.ckpt_dir is not None else None,
        "ckpt_prefix": args.ckpt_prefix,
        "policy_checkpoint_status": ckpt_status,
        "allow_random_policy_init": bool(args.allow_random_policy_init),
        "device": str(device),
        "datagen_mode": args.datagen_mode,
        "pool_size": int(args.pool_size),
        "eval_pool_size": int(eval_pool_size),
        "group_size": int(args.group_size),
        "rollout_steps": int(args.rollout_steps),
        "eval_rollout_steps": int(eval_rollout_steps),
        "branch_num": int(args.branch_num),
        "batch_size": int(args.batch_size),
        "metric_batch_cap": int(args.metric_batch_cap),
        "n_train_batches": int(len(train_batches)),
        "n_eval_batches": int(len(eval_batches)),
        "n_metric_batches": int(len(metric_batches)),
        "q_stage_epochs": int(args.q_stage_epochs),
        "pvbp_stage_epochs": int(args.pvbp_stage_epochs),
        "q_refresh_stage_epochs": int(args.q_refresh_stage_epochs),
        "eval_every": int(args.eval_every),
        "train_generation": train_result.generation_summary,
        "eval_generation": eval_result.generation_summary,
        "history": history,
    }
    with open(args.output_dir / "alignment_summary.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    _write_history_csv(history, args.output_dir / "alignment_history.csv")
    _plot_history(history, args.output_dir)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
