"""
Compare two micro-data generation schemes for policy/value training.

Goal:
- hold the model architecture / loss / optimizer family fixed
- change only how Bellman parent tuples are generated
- test whether replaying promoted children as future parents improves
  Bellman convergence relative to the current main-branch rollout

Schemes:
1. main_branch:
   - exactly mirrors current modeb rollout logic
   - every parent expands to all children
   - only main_branch child becomes the next parent

2. children_promote:
   - every parent expands to all children
   - next parent pool is sampled from all children across the current pool
   - pool size stays fixed; no full tree explosion
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from data import SimulateTS, TensorTable  # noqa: E402
from data.tensor_data import cat_rows  # noqa: E402
from experiments.run_utils import build_hyperparams, build_models, build_optimizers  # noqa: E402
from training.episode import Episode  # noqa: E402


@dataclass
class DatagenRunResult:
    name: str
    firm_table: TensorTable
    macro_table: TensorTable
    generation_summary: Dict[str, Any]


@dataclass
class ArmRunContext:
    name: str
    datagen_result: DatagenRunResult
    episode: Episode
    batches: List[Dict[str, torch.Tensor]]
    conv_before: Dict[str, Any]
    common_before: Dict[str, Dict[str, Any]]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_policy_hyperparams(
    n_epochs: int,
    q_stage_epochs: int,
    pvbp_stage_epochs: int,
    q_refresh_stage_epochs: int,
) -> Any:
    hp = build_hyperparams()
    hp.q_stage_epochs = int(q_stage_epochs)
    hp.pvbp_stage_epochs = int(pvbp_stage_epochs)
    hp.q_refresh_stage_epochs = int(q_refresh_stage_epochs)
    hp.q_pretrain_epochs = int(q_stage_epochs)
    hp.q_warmstart_epochs = int(q_stage_epochs)
    hp.epochs = int(max(n_epochs, q_stage_epochs + pvbp_stage_epochs + q_refresh_stage_epochs))
    return hp


def _snapshot_model_states(models: Dict[str, torch.nn.Module]) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        name: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for name, model in models.items()
    }


def _load_model_states(
    models: Dict[str, torch.nn.Module],
    states: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device,
) -> None:
    for name, state in states.items():
        if name not in models:
            raise KeyError(f"Initial state contains unknown model: {name}")
        state_on_device = {k: v.to(device=device) for k, v in state.items()}
        models[name].load_state_dict(state_on_device, strict=True)


def _model_state_stats(states: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for model_name, state in states.items():
        n_params = 0
        abs_sum = 0.0
        sq_sum = 0.0
        max_abs = 0.0
        for tensor in state.values():
            vals = tensor.detach().to(dtype=torch.float64).reshape(-1)
            n_params += int(vals.numel())
            if vals.numel() == 0:
                continue
            abs_vals = vals.abs()
            abs_sum += float(abs_vals.sum().item())
            sq_sum += float((vals * vals).sum().item())
            max_abs = max(max_abs, float(abs_vals.max().item()))
        out[model_name] = {
            "n_params": float(n_params),
            "abs_sum": abs_sum,
            "l2": float(sq_sum ** 0.5),
            "max_abs": max_abs,
        }
    return out


def _policy_checkpoint_status(ckpt_dir: Path | None, ckpt_prefix: str | None) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "ckpt_dir": str(ckpt_dir) if ckpt_dir is not None else None,
        "ckpt_prefix": ckpt_prefix,
        "policy_value_loaded": False,
        "policy_value_source": None,
        "reason": None,
    }
    if ckpt_dir is None:
        status["reason"] = "no_ckpt_dir"
        return status

    ckpt_dir = Path(ckpt_dir)
    prefix = f"{ckpt_prefix}_" if ckpt_prefix else ""
    split_q = ckpt_dir / f"{prefix}policy_value_q.pt"
    split_pvbp = ckpt_dir / f"{prefix}policy_value_pvbp.pt"
    merged = ckpt_dir / f"{prefix}policy_value.pt"
    status.update(
        {
            "split_q_exists": split_q.exists(),
            "split_pvbp_exists": split_pvbp.exists(),
            "merged_exists": merged.exists(),
        }
    )
    if split_q.exists() and split_pvbp.exists():
        status["policy_value_loaded"] = True
        status["policy_value_source"] = "split_q_pvbp"
        status["reason"] = "loaded_split_policy_value"
        return status
    if merged.exists():
        try:
            state = torch.load(merged, map_location="cpu")
            is_old_joint_ckpt = any(
                str(k).startswith("shared_model.") or str(k).startswith("combined_model.")
                for k in state.keys()
            )
            status["merged_legacy_joint"] = bool(is_old_joint_ckpt)
            if is_old_joint_ckpt:
                status["reason"] = "merged_policy_value_is_incompatible_legacy_joint"
            else:
                status["policy_value_loaded"] = True
                status["policy_value_source"] = "merged_policy_value"
                status["reason"] = "loaded_merged_policy_value"
        except Exception as exc:
            status["reason"] = f"merged_policy_value_inspect_failed: {exc}"
        return status

    status["reason"] = "missing_policy_value_checkpoint"
    return status


def _build_episode_from_initial_state(
    initial_model_states: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device,
    hp: Any,
) -> Episode:
    models = build_models(device=device)
    _load_model_states(models, initial_model_states, device)
    optimizers = build_optimizers(models, hp)
    return Episode(
        models=models,
        optimizers=optimizers,
        config=Config,
        hyperparams=hp,
        device=device,
        episode_id=0,
    )


def _clone_state(state: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            out[key] = value.clone()
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _init_parent_pool(sim: SimulateTS, pool_size: int, seed: int) -> List[Dict[str, Any]]:
    _set_seed(seed)
    return [sim._initialize_path_tensor(i) for i in range(pool_size)]


def _summarize_generation(
    mode: str,
    pool_size: int,
    rollout_steps: int,
    promoted_branch_counts: Sequence[int],
    promoted_from_main_count: int,
    promoted_total: int,
    alive_parent_counts: Sequence[int],
    alive_child_counts: Sequence[int],
) -> Dict[str, Any]:
    branch_share = {}
    total_branch = max(1, int(sum(promoted_branch_counts)))
    for branch_k, cnt in enumerate(promoted_branch_counts):
        branch_share[f"branch_{branch_k}_share"] = float(cnt / total_branch)
    return {
        "mode": mode,
        "branch_num": int(len(promoted_branch_counts)),
        "pool_size": int(pool_size),
        "rollout_steps": int(rollout_steps),
        "promoted_total": int(promoted_total),
        "promoted_from_main_share": float(promoted_from_main_count / max(1, promoted_total)),
        "alive_parent_mean": float(np.mean(alive_parent_counts)) if alive_parent_counts else 0.0,
        "alive_parent_p90": float(np.quantile(alive_parent_counts, 0.9)) if alive_parent_counts else 0.0,
        "alive_child_mean": float(np.mean(alive_child_counts)) if alive_child_counts else 0.0,
        "alive_child_p90": float(np.quantile(alive_child_counts, 0.9)) if alive_child_counts else 0.0,
        **branch_share,
    }


def generate_policy_tuple_tables(
    sim: SimulateTS,
    initial_parent_pool: Sequence[Dict[str, Any]],
    rollout_steps: int,
    mode: str,
    seed: int,
) -> DatagenRunResult:
    if mode not in {"main_branch", "children_promote"}:
        raise ValueError(f"Unknown mode: {mode}")

    rng = np.random.default_rng(seed)
    parent_pool = [_clone_state(s) for s in initial_parent_pool]
    pool_size = len(parent_pool)
    firm_rows: List[torch.Tensor] = []
    macro_rows: List[torch.Tensor] = []
    promoted_branch_counts = [0 for _ in range(sim.branch_num)]
    promoted_from_main_count = 0
    promoted_total = 0
    alive_parent_counts: List[int] = []
    alive_child_counts: List[int] = []

    for t in range(rollout_steps):
        all_children: List[Tuple[int, int, Dict[str, Any]]] = []
        main_children: List[Tuple[int, int, Dict[str, Any]]] = []

        for slot_idx, parent_state_in in enumerate(parent_pool):
            parent_state = _clone_state(parent_state_in)
            alive_parent_counts.append(int(parent_state["alive"].sum().item()))

            parent_firm, parent_macro = sim._process_node_tensor(parent_state, slot_idx, t, branch_k=-1)
            firm_rows.append(parent_firm)
            macro_rows.append(parent_macro)

            branch_states = sim._expand_branches_tensor(parent_state)
            processed_children: List[Tuple[int, int, Dict[str, Any]]] = []
            for branch_k, branch_state in enumerate(branch_states):
                if sim.enable_entry:
                    branch_state = sim._apply_entry_tensor(branch_state)
                    branch_states[branch_k] = branch_state

                branch_firm, branch_macro = sim._process_node_tensor(branch_state, slot_idx, t + 1, branch_k=branch_k)
                firm_rows.append(branch_firm)
                macro_rows.append(branch_macro)

                alive_child_counts.append(int(branch_state["alive"].sum().item()))
                processed_children.append((slot_idx, branch_k, _clone_state(branch_state)))

            all_children.extend(processed_children)
            main_children.append(processed_children[sim.main_branch])

        if mode == "main_branch":
            selected_children = main_children
        else:
            replace = len(all_children) < pool_size
            picked_idx = rng.choice(len(all_children), size=pool_size, replace=replace)
            selected_children = [all_children[int(i)] for i in picked_idx]

        next_parent_pool: List[Dict[str, Any]] = []
        for _, branch_k, child_state_in in selected_children:
            promoted_branch_counts[branch_k] += 1
            promoted_total += 1
            if branch_k == sim.main_branch:
                promoted_from_main_count += 1

            next_state = _clone_state(child_state_in)
            if sim.enable_exit:
                next_state = sim._apply_exit(next_state)
            next_parent_pool.append(next_state)

        parent_pool = next_parent_pool

    firm_tensor = cat_rows(firm_rows, len(sim.FIRM_COLUMNS), sim.device)
    macro_tensor = cat_rows(macro_rows, len(sim.MACRO_COLUMNS), sim.device)
    generation_summary = _summarize_generation(
        mode=mode,
        pool_size=pool_size,
        rollout_steps=rollout_steps,
        promoted_branch_counts=promoted_branch_counts,
        promoted_from_main_count=promoted_from_main_count,
        promoted_total=promoted_total,
        alive_parent_counts=alive_parent_counts,
        alive_child_counts=alive_child_counts,
    )
    return DatagenRunResult(
        name=mode,
        firm_table=TensorTable(firm_tensor, sim.FIRM_COLUMNS),
        macro_table=TensorTable(macro_tensor, sim.MACRO_COLUMNS),
        generation_summary=generation_summary,
    )


def _extract_conv(convergence: Dict[str, Any] | None) -> Dict[str, float]:
    eq = (convergence or {}).get("equations", {})
    out: Dict[str, float] = {}
    for name in ("p0", "pi", "q"):
        item = eq.get(name, {})
        out[f"conv_{name}_mean"] = float(item.get("mean", float("nan")))
        out[f"conv_{name}_p90"] = float(item.get("p90", float("nan")))
        out[f"conv_{name}_passed"] = float(1.0 if item.get("passed", False) else 0.0)
    out["convergence_passed"] = float(1.0 if (convergence or {}).get("passed", False) else 0.0)
    return out


def _path_col_idx(table: TensorTable) -> int:
    return list(table.columns).index("path")


def _next_path_offset(*tables: TensorTable) -> int:
    max_path = -1
    for table in tables:
        if len(table) == 0:
            continue
        path_idx = _path_col_idx(table)
        max_path = max(max_path, int(table.data[:, path_idx].max().item()))
    return max_path + 1


def _offset_path_column(table: TensorTable, offset: int) -> TensorTable:
    data = table.data.clone()
    if data.numel() > 0 and offset != 0:
        data[:, _path_col_idx(table)] += float(offset)
    return TensorTable(data, list(table.columns))


def _prefix_dict(prefix: str, values: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in values.items()}


def _extract_stage_final_losses(summary: Dict[str, Any]) -> Dict[str, float]:
    final_losses = summary.get("final_losses", {})
    out: Dict[str, float] = {}
    for key, value in final_losses.items():
        try:
            out[key] = float(value)
        except Exception:
            continue
    return out


def _plot_conv_compare(summary_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    labels = [row["name"] for row in summary_rows]
    metrics = [
        "train_support_conv_p0_mean_after",
        "train_support_conv_pi_mean_after",
        "train_support_conv_q_mean_after",
    ]
    x = np.arange(len(labels))
    width = 0.22
    plt.figure(figsize=(7.2, 4.2))
    for idx, metric in enumerate(metrics):
        vals = [row.get(metric, np.nan) for row in summary_rows]
        plt.bar(x + (idx - 1) * width, vals, width=width, label=metric.replace("_after", ""))
    plt.xticks(x, labels)
    plt.ylabel("absolute Bellman residual mean")
    plt.title("Policy datagen compare: post-train convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_branch_share(summary_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    branch_keys = sorted([k for k in summary_rows[0].keys() if k.startswith("branch_") and k.endswith("_share")])
    labels = [row["name"] for row in summary_rows]
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels), dtype=float)
    plt.figure(figsize=(6.8, 4.0))
    for key in branch_keys:
        vals = np.array([float(row.get(key, 0.0)) for row in summary_rows], dtype=float)
        plt.bar(x, vals, bottom=bottom, label=key.replace("_share", ""))
        bottom += vals
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("promotion share")
    plt.title("Promoted child branch composition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_common_eval_compare(summary_rows: Sequence[Dict[str, Any]], eval_name: str, out_path: Path) -> None:
    labels = [row["name"] for row in summary_rows]
    metrics = [
        f"{eval_name}_conv_p0_mean_after",
        f"{eval_name}_conv_pi_mean_after",
        f"{eval_name}_conv_q_mean_after",
    ]
    x = np.arange(len(labels))
    width = 0.22
    plt.figure(figsize=(7.2, 4.2))
    for idx, metric in enumerate(metrics):
        vals = [row.get(metric, np.nan) for row in summary_rows]
        plt.bar(x + (idx - 1) * width, vals, width=width, label=metric.replace(f"{eval_name}_", "").replace("_after", ""))
    plt.xticks(x, labels)
    plt.ylabel("absolute Bellman residual mean")
    plt.title(f"Policy datagen compare: {eval_name} common eval")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_stage_common_eval(summary_rows: Sequence[Dict[str, Any]], eval_name: str, out_path: Path) -> None:
    phases = ["q_stage_end", "pvbp_stage_end", "q_refresh_end"]
    equations = ["p0", "pi", "q"]
    fig, axes = plt.subplots(1, len(equations), figsize=(13.5, 3.8), sharex=True)
    for ax, eq in zip(axes, equations):
        for row in summary_rows:
            vals = [
                row.get(f"stage_{phase}_{eval_name}_conv_{eq}_mean", np.nan)
                for phase in phases
            ]
            ax.plot(phases, vals, marker="o", label=row["name"])
        ax.set_title(f"{eq.upper()} mean residual")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("absolute residual mean")
    axes[-1].legend()
    fig.suptitle(f"Stage-level common eval: {eval_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def prepare_single_arm(
    name: str,
    datagen_result: DatagenRunResult,
    device: torch.device,
    batch_size: int,
    n_epochs: int,
    seed: int,
    q_stage_epochs: int,
    pvbp_stage_epochs: int,
    q_refresh_stage_epochs: int,
    common_eval_batches: Dict[str, List[Dict[str, torch.Tensor]]],
    initial_model_states: Dict[str, Dict[str, torch.Tensor]],
) -> ArmRunContext:
    _set_seed(seed)
    hp = _make_policy_hyperparams(
        n_epochs=n_epochs,
        q_stage_epochs=q_stage_epochs,
        pvbp_stage_epochs=pvbp_stage_epochs,
        q_refresh_stage_epochs=q_refresh_stage_epochs,
    )
    episode = _build_episode_from_initial_state(initial_model_states, device, hp)

    _set_seed(seed)
    batches = episode._create_firm_batches_from_tensor(
        datagen_result.firm_table,
        batch_size=batch_size,
        n_branches=int(datagen_result.generation_summary.get("branch_num", 2) or 2),
        eta_resample=True,
    )
    if not batches:
        raise RuntimeError(f"No training batches created for arm: {name}")

    conv_before = episode.evaluate_bellman_convergence(batches)
    common_before = {
        eval_name: episode.evaluate_bellman_convergence(eval_batches)
        for eval_name, eval_batches in common_eval_batches.items()
    }
    return ArmRunContext(
        name=name,
        datagen_result=datagen_result,
        episode=episode,
        batches=batches,
        conv_before=conv_before,
        common_before=common_before,
    )


def _assert_common_before_matches(
    contexts: Sequence[ArmRunContext],
    eval_names: Sequence[str],
    tol: float,
) -> Dict[str, Any]:
    if len(contexts) < 2:
        return {"enabled": False, "reason": "fewer_than_two_arms"}

    ref = contexts[0]
    comparisons: List[Dict[str, Any]] = []
    max_abs_diff = 0.0
    worst: Dict[str, Any] | None = None
    metric_keys = [
        "conv_p0_mean",
        "conv_p0_p90",
        "conv_pi_mean",
        "conv_pi_p90",
        "conv_q_mean",
        "conv_q_p90",
    ]
    for ctx in contexts[1:]:
        for eval_name in eval_names:
            ref_values = _extract_conv(ref.common_before.get(eval_name, {}))
            cur_values = _extract_conv(ctx.common_before.get(eval_name, {}))
            for key in metric_keys:
                a = float(ref_values.get(key, float("nan")))
                b = float(cur_values.get(key, float("nan")))
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue
                diff = abs(a - b)
                item = {
                    "reference_arm": ref.name,
                    "other_arm": ctx.name,
                    "eval_name": eval_name,
                    "metric": key,
                    "reference": a,
                    "other": b,
                    "abs_diff": diff,
                }
                comparisons.append(item)
                if diff > max_abs_diff:
                    max_abs_diff = diff
                    worst = item

    check = {
        "enabled": True,
        "tolerance": float(tol),
        "max_abs_diff": float(max_abs_diff),
        "passed": bool(max_abs_diff <= tol),
        "worst": worst,
        "comparisons": comparisons,
    }
    if max_abs_diff > tol:
        raise RuntimeError(
            "Common eval before residuals differ across arms. "
            f"max_abs_diff={max_abs_diff:.6e}, tol={tol:.6e}, worst={worst}. "
            "This violates clean A/B initialization; aborting before training."
        )
    return check


def train_prepared_arm(
    context: ArmRunContext,
    n_epochs: int,
    log_interval: int,
    common_eval_batches: Dict[str, List[Dict[str, torch.Tensor]]],
) -> Dict[str, Any]:
    name = context.name
    datagen_result = context.datagen_result
    episode = context.episode
    batches = context.batches
    conv_before = context.conv_before
    common_before = context.common_before

    stage_common_eval: Dict[str, Dict[str, float]] = {}

    def _on_policy_stage(stage_summary: Dict[str, Any]) -> None:
        phase = str(stage_summary.get("phase", "unknown"))
        for eval_name, eval_batches in common_eval_batches.items():
            conv = episode.evaluate_bellman_convergence(eval_batches)
            stage_common_eval.update(_prefix_dict(f"stage_{phase}_{eval_name}", _extract_conv(conv)))

    train_summary = episode._run_batches(
        batches=batches,
        n_epochs=n_epochs,
        log_interval=log_interval,
        train_modules=["policy_value"],
        desc_prefix=f"{name} ",
        policy_stage_callback=_on_policy_stage,
    )
    conv_after = train_summary.get("convergence", {})
    common_after = {
        eval_name: episode.evaluate_bellman_convergence(eval_batches)
        for eval_name, eval_batches in common_eval_batches.items()
    }

    out = {
        "name": name,
        "n_firm_rows": int(len(datagen_result.firm_table)),
        "n_macro_rows": int(len(datagen_result.macro_table)),
        "n_batches": int(len(batches)),
        **datagen_result.generation_summary,
        **_prefix_dict("train_support", {f"{k}_before": v for k, v in _extract_conv(conv_before).items()}),
        **_prefix_dict("train_support", {f"{k}_after": v for k, v in _extract_conv(conv_after).items()}),
        **_extract_stage_final_losses(train_summary),
    }
    for eval_name, conv in common_before.items():
        out.update(_prefix_dict(f"{eval_name}", {f"{k}_before": v for k, v in _extract_conv(conv).items()}))
    for eval_name, conv in common_after.items():
        out.update(_prefix_dict(f"{eval_name}", {f"{k}_after": v for k, v in _extract_conv(conv).items()}))
    out.update(stage_common_eval)
    return out


def _make_eval_batches(
    table: TensorTable,
    device: torch.device,
    batch_size: int,
    branch_num: int,
    seed: int,
) -> List[Dict[str, torch.Tensor]]:
    _set_seed(seed)
    models = build_models(device=device)
    hp = build_hyperparams()
    optimizers = build_optimizers(models, hp)
    episode = Episode(
        models=models,
        optimizers=optimizers,
        config=Config,
        hyperparams=hp,
        device=device,
        episode_id=0,
    )
    return episode._create_firm_batches_from_tensor(
        table,
        batch_size=batch_size,
        n_branches=branch_num,
        eta_resample=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare policy/value micro training under different tuple data generation schemes.")
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--ckpt-prefix", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--branch-num", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=220)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q-stage-epochs", type=int, default=100)
    parser.add_argument("--pvbp-stage-epochs", type=int, default=100)
    parser.add_argument("--q-refresh-stage-epochs", type=int, default=20)
    parser.add_argument("--eval-pool-size", type=int, default=None)
    parser.add_argument("--eval-rollout-steps", type=int, default=None)
    parser.add_argument("--before-eval-tol", type=float, default=1e-8)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = ROOT / "experiments" / "policy_datagen_compare"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    _set_seed(args.seed)
    base_models = build_models(device=device, ckpt_dir=args.ckpt_dir, ckpt_prefix=args.ckpt_prefix)
    initial_model_states = _snapshot_model_states(base_models)
    initial_state_stats = _model_state_stats(initial_model_states)
    policy_ckpt_status = _policy_checkpoint_status(args.ckpt_dir, args.ckpt_prefix)
    sim = SimulateTS(
        models=base_models,
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
    initial_parent_pool = _init_parent_pool(sim, pool_size=args.pool_size, seed=args.seed)

    main_result = generate_policy_tuple_tables(
        sim=sim,
        initial_parent_pool=initial_parent_pool,
        rollout_steps=args.rollout_steps,
        mode="main_branch",
        seed=args.seed,
    )
    promote_result = generate_policy_tuple_tables(
        sim=sim,
        initial_parent_pool=initial_parent_pool,
        rollout_steps=args.rollout_steps,
        mode="children_promote",
        seed=args.seed + 1,
    )
    eval_pool_size = int(args.pool_size if args.eval_pool_size is None else args.eval_pool_size)
    eval_rollout_steps = int(args.rollout_steps if args.eval_rollout_steps is None else args.eval_rollout_steps)
    eval_parent_pool = _init_parent_pool(sim, pool_size=eval_pool_size, seed=args.seed + 9000)
    eval_main_result = generate_policy_tuple_tables(
        sim=sim,
        initial_parent_pool=eval_parent_pool,
        rollout_steps=eval_rollout_steps,
        mode="main_branch",
        seed=args.seed + 9001,
    )
    eval_children_result = generate_policy_tuple_tables(
        sim=sim,
        initial_parent_pool=eval_parent_pool,
        rollout_steps=eval_rollout_steps,
        mode="children_promote",
        seed=args.seed + 9002,
    )
    eval_children_path_offset = _next_path_offset(eval_main_result.firm_table, eval_main_result.macro_table)
    eval_children_firm_offset = _offset_path_column(eval_children_result.firm_table, eval_children_path_offset)
    eval_children_macro_offset = _offset_path_column(eval_children_result.macro_table, eval_children_path_offset)
    eval_mixed_tensor = torch.cat([eval_main_result.firm_table.data, eval_children_firm_offset.data], dim=0)
    eval_mixed_result = DatagenRunResult(
        name="eval_mixed",
        firm_table=TensorTable(eval_mixed_tensor, sim.FIRM_COLUMNS),
        macro_table=TensorTable(
            torch.cat([eval_main_result.macro_table.data, eval_children_macro_offset.data], dim=0),
            sim.MACRO_COLUMNS,
        ),
        generation_summary={
            "mode": "eval_mixed",
            "branch_num": int(args.branch_num),
            "eval_children_path_offset": int(eval_children_path_offset),
            "source_supports": ["eval_main", "eval_children"],
        },
    )
    common_eval_batches = {
        "eval_main": _make_eval_batches(
            eval_main_result.firm_table,
            device,
            args.batch_size,
            args.branch_num,
            seed=args.seed + 9101,
        ),
        "eval_children": _make_eval_batches(
            eval_children_result.firm_table,
            device,
            args.batch_size,
            args.branch_num,
            seed=args.seed + 9102,
        ),
        "eval_mixed": _make_eval_batches(
            eval_mixed_result.firm_table,
            device,
            args.batch_size,
            args.branch_num,
            seed=args.seed + 9103,
        ),
    }

    arm_contexts = [
        prepare_single_arm(
            name="main_branch",
            datagen_result=main_result,
            device=device,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            seed=args.seed + 100,
            q_stage_epochs=args.q_stage_epochs,
            pvbp_stage_epochs=args.pvbp_stage_epochs,
            q_refresh_stage_epochs=args.q_refresh_stage_epochs,
            common_eval_batches=common_eval_batches,
            initial_model_states=initial_model_states,
        ),
        prepare_single_arm(
            name="children_promote",
            datagen_result=promote_result,
            device=device,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            seed=args.seed + 101,
            q_stage_epochs=args.q_stage_epochs,
            pvbp_stage_epochs=args.pvbp_stage_epochs,
            q_refresh_stage_epochs=args.q_refresh_stage_epochs,
            common_eval_batches=common_eval_batches,
            initial_model_states=initial_model_states,
        ),
    ]
    before_eval_check = _assert_common_before_matches(
        contexts=arm_contexts,
        eval_names=list(common_eval_batches.keys()),
        tol=float(args.before_eval_tol),
    )

    summary_rows = [
        train_prepared_arm(
            context=ctx,
            n_epochs=args.n_epochs,
            log_interval=args.log_interval,
            common_eval_batches=common_eval_batches,
        )
        for ctx in arm_contexts
    ]

    manifest = {
        "ckpt_dir": str(args.ckpt_dir) if args.ckpt_dir is not None else None,
        "ckpt_prefix": args.ckpt_prefix,
        "device": str(device),
        "pool_size": int(args.pool_size),
        "group_size": int(args.group_size),
        "rollout_steps": int(args.rollout_steps),
        "branch_num": int(args.branch_num),
        "batch_size": int(args.batch_size),
        "n_epochs": int(args.n_epochs),
        "q_stage_epochs": int(args.q_stage_epochs),
        "pvbp_stage_epochs": int(args.pvbp_stage_epochs),
        "q_refresh_stage_epochs": int(args.q_refresh_stage_epochs),
        "eval_pool_size": int(eval_pool_size),
        "eval_rollout_steps": int(eval_rollout_steps),
        "before_eval_tol": float(args.before_eval_tol),
        "initialization": {
            "shared_initial_model_state": True,
            "policy_checkpoint_status": policy_ckpt_status,
            "state_stats": initial_state_stats,
            "before_eval_check": before_eval_check,
        },
        "eval_supports": {
            "eval_main": eval_main_result.generation_summary,
            "eval_children": eval_children_result.generation_summary,
            "eval_mixed": eval_mixed_result.generation_summary,
        },
        "results": summary_rows,
    }
    with open(args.output_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    _plot_conv_compare(summary_rows, args.output_dir / "compare_conv_means.png")
    _plot_branch_share(summary_rows, args.output_dir / "compare_promoted_branch_share.png")
    for eval_name in common_eval_batches:
        _plot_common_eval_compare(summary_rows, eval_name, args.output_dir / f"compare_{eval_name}_conv_means.png")
        _plot_stage_common_eval(summary_rows, eval_name, args.output_dir / f"compare_{eval_name}_stage_conv_means.png")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
