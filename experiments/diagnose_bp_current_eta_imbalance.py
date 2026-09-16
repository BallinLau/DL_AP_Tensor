from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis.bp_eta_imbalance import (  # noqa: E402
    audit_bp_cache_by_current_eta,
    bp_teacher_cache_sha256,
    configure_bp_heads_only,
    eta_count_summary,
    eta_switch_panel_sensitivity,
    flatten_bp_target_cache,
    gradient_contribution_audit,
    loss_contribution_audit,
    parameter_change_summary,
    sample_current_eta_rows,
    select_bp_cache_rows,
    state_dict_sha256,
    training_equivalent_total_objective_audit,
)
from analysis.checkpoint_loader import load_analysis_checkpoint  # noqa: E402
from config import Config  # noqa: E402
from training.bp_grid_teacher import BPGridTeacher  # noqa: E402
from training.episode import Episode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated BP-head-only experiments that vary current parent eta_t "
            "sampling while keeping the economic transition and teacher cache fixed."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--firm-data", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=7)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cache-batch-size", type=int, default=4096)
    parser.add_argument(
        "--max-cache-rows",
        type=int,
        default=0,
        help="Optional diagnostic cap; 0 audits the complete formal BP cache.",
    )
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--eta1-train-share",
        nargs="+",
        default=["natural", "0.25", "0.50"],
        help="Controlled samplers to run: natural, 0.25, and/or 0.50.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--freeze-shared-trunk",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow-current-config", action="store_true")
    return parser.parse_args()


@contextmanager
def checkpoint_economic_config(economic_config: Any):
    saved = {name: getattr(Config, name) for name in economic_config.field_names()}
    try:
        for name, value in economic_config.to_dict().items():
            setattr(Config, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(Config, name, value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_experiments(values: Iterable[str]) -> tuple[tuple[str, Optional[float]], ...]:
    lookup = {"natural": ("A_natural", None), "0.25": ("B_eta25", 0.25), "0.50": ("C_eta50", 0.50), "0.5": ("C_eta50", 0.50)}
    resolved = []
    for raw in values:
        key = str(raw).strip().lower()
        if key not in lookup:
            raise ValueError(f"Unsupported --eta1-train-share value: {raw!r}")
        if lookup[key] not in resolved:
            resolved.append(lookup[key])
    return tuple(resolved)


def _slice_batch(batch: Mapping[str, Any], count: int) -> Dict[str, Any]:
    n_rows = int(batch["parent"].shape[0])
    result: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == n_rows:
            result[key] = value[:count]
        elif isinstance(value, list):
            result[key] = [item[:count] for item in value]
        else:
            result[key] = value
    return result


def truncate_batches(batches: Iterable[Mapping[str, Any]], max_rows: int) -> list[Dict[str, Any]]:
    batches = list(batches)
    if max_rows <= 0:
        return [dict(batch) for batch in batches]
    selected: list[Dict[str, Any]] = []
    remaining = int(max_rows)
    for batch in batches:
        if remaining <= 0:
            break
        count = min(remaining, int(batch["parent"].shape[0]))
        if count > 0:
            selected.append(_slice_batch(batch, count))
            remaining -= count
    return selected


def make_episode(
    online_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    hyperparams: Any,
    device: torch.device,
    episode_id: int,
) -> Episode:
    optimizer = torch.optim.AdamW(online_model.parameters(), lr=1e-6)
    return Episode(
        models={"policy_value": online_model},
        optimizers={"policy_value": optimizer},
        config=Config,
        hyperparams=hyperparams,
        device=device,
        episode_id=episode_id,
        firm_target=teacher_model,
    )


@torch.no_grad()
def evaluate_exact_regrets(
    episode: Episode,
    batches: list[Dict[str, Any]],
    teacher_model: torch.nn.Module,
    policy_model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    """Re-evaluate regret at current BP predictions under the fixed teacher."""
    teacher = BPGridTeacher.from_hyperparams(
        teacher_model,
        episode.loss_fns["p0"],
        episode.loss_fns["pi"],
        episode.hyperparams,
    )
    teacher_model.eval()
    policy_model.eval()
    result = {"bp0": [], "bpi": []}
    for batch in batches:
        parent = batch["parent"]
        children = batch.get("children", [])
        if not children:
            children = [batch["child0"], batch["child1"]]
        parent_state = parent[:, :7] if parent.shape[1] > 7 else parent
        raw_m, m_list = episode._build_policy_m_lists(
            parent,
            children,
            float(getattr(episode.hyperparams, "pv_m_clamp_min", 0.7)),
            float(getattr(episode.hyperparams, "pv_m_clamp_max", 1.3)),
        )
        expanded_children, _, expanded_m, child_weights = episode._expand_policy_expectation_children(
            children,
            raw_m,
            m_list,
        )
        bp0, bpi = policy_model.forward_policy(parent_state)
        for name, branch, prediction in (("bp0", "p0", bp0), ("bpi", "pi", bpi)):
            teacher_result = teacher.compute(
                parent_state=parent_state,
                children=expanded_children,
                m_list=expanded_m,
                branch=branch,
                bp_pred=prediction,
                child_weights=child_weights,
            )
            result[name].append(teacher_result["regret"].detach().cpu())
    return {key: torch.cat(values, dim=0) for key, values in result.items()}


def _metric_rows(
    *,
    experiment: str,
    step: int,
    model: torch.nn.Module,
    cache: Mapping[str, torch.Tensor],
    regrets: Mapping[str, torch.Tensor],
    train_eta0_share: float,
    train_eta1_share: float,
) -> list[Dict[str, Any]]:
    rows = audit_bp_cache_by_current_eta(model, cache, regrets=regrets)
    for row in rows:
        row.update({
            "experiment": experiment,
            "step": int(step),
            "train_eta0_share": float(train_eta0_share),
            "train_eta1_share": float(train_eta1_share),
        })
    return rows


def run_experiment(
    *,
    name: str,
    eta1_share: Optional[float],
    base_model: torch.nn.Module,
    base_episode: Episode,
    teacher_model: torch.nn.Module,
    batches: list[Dict[str, Any]],
    cache: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
    output_dir: Path,
    initial_head_hash: str,
    teacher_cache_hash: str,
) -> tuple[pd.DataFrame, Dict[str, Any], torch.nn.Module]:
    set_seed(args.seed)
    model = copy.deepcopy(base_model).to(base_episode.device)
    if state_dict_sha256(model, ("bp0_head.", "bpi_head.")) != initial_head_hash:
        raise RuntimeError("Experiment did not start from the shared initial BP-head state")
    trainable = configure_bp_heads_only(model)
    before = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    runner = copy.copy(base_episode)
    runner.models = dict(base_episode.models)
    runner.models["policy_value"] = model
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    eta = cache["eta_current"].reshape(-1).bool()

    metrics: list[Dict[str, Any]] = []
    sampled_eta0 = 0
    sampled_eta1 = 0

    def _record(step: int) -> None:
        regrets = evaluate_exact_regrets(runner, batches, teacher_model, model)
        model.eval()
        with torch.no_grad():
            overall_loss, _ = runner._compute_bp_cache_loss(
                select_bp_cache_rows(
                    cache,
                    torch.arange(cache["parent"].shape[0], dtype=torch.long),
                )
            )
        total = sampled_eta0 + sampled_eta1
        natural_eta1 = float(eta.float().mean().item())
        train_eta1 = sampled_eta1 / total if total else (natural_eta1 if eta1_share is None else float(eta1_share))
        record_rows = _metric_rows(
            experiment=name,
            step=step,
            model=model,
            cache=cache,
            regrets=regrets,
            train_eta0_share=1.0 - train_eta1,
            train_eta1_share=train_eta1,
        )
        for row in record_rows:
            row["overall_bp_loss"] = float(overall_loss.item())
        metrics.extend(record_rows)

    _record(0)
    for step in range(1, int(args.steps) + 1):
        row_index = sample_current_eta_rows(
            cache["eta_current"],
            batch_size=int(args.batch_size),
            eta1_share=eta1_share,
            generator=generator,
        )
        sampled_eta1 += int(eta[row_index].sum().item())
        sampled_eta0 += int(row_index.numel()) - int(eta[row_index].sum().item())
        item = select_bp_cache_rows(cache, row_index)
        # Keep the frozen representation deterministic even if a checkpoint
        # was built with non-zero dropout; only the two output heads train.
        model.eval()
        model.bp0_head.train()
        model.bpi_head.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = runner._compute_bp_cache_loss(item)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite BP loss in {name} at step {step}")
        loss.backward()
        optimizer.step()
        if step % max(1, int(args.record_every)) == 0 or step == int(args.steps):
            _record(step)

    changes = parameter_change_summary(before, model)
    if changes["non_bp_parameter_max_change"] != 0.0:
        raise RuntimeError(f"{name} changed frozen non-BP parameters")
    if changes["bp_head_parameter_max_change"] <= 0.0:
        raise RuntimeError(f"{name} did not update BP-head parameters")
    frame = pd.DataFrame(metrics)
    final_rows = frame.loc[frame["step"] == frame["step"].max()]
    final: Dict[str, Any] = {
        "experiment": name,
        "requested_eta1_train_share": "natural" if eta1_share is None else eta1_share,
        "realized_eta1_train_share": sampled_eta1 / max(sampled_eta0 + sampled_eta1, 1),
        "optimizer_steps": int(args.steps),
        "seed": int(args.seed),
        "initial_bp_head_hash": initial_head_hash,
        "teacher_cache_hash": teacher_cache_hash,
        "final_bp_head_hash": state_dict_sha256(model, ("bp0_head.", "bpi_head.")),
        **changes,
    }
    for _, row in final_rows.iterrows():
        prefix = f"{row['branch']}_eta{int(row['eta_current'])}"
        for field in ("bp_pred_mean", "bp_star_mean", "mae", "mse", "mean_regret"):
            final[f"{prefix}_{field}"] = float(row[field])
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "final_summary.json").write_text(
        json.dumps(final, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    return frame, final, model


def plot_comparisons(metrics: pd.DataFrame, output_dir: Path) -> None:
    colors = {"A_natural": "#1f77b4", "B_eta25": "#ff7f0e", "C_eta50": "#2ca02c"}
    for branch in ("bp0", "bpi"):
        for eta_value in (0, 1):
            for metric, ylabel in (("bp_pred_mean", "Mean BP prediction"), ("mae", "BP MAE")):
                fig, ax = plt.subplots(figsize=(7.2, 4.6))
                subset = metrics.loc[
                    (metrics["branch"] == branch) & (metrics["eta_current"] == eta_value)
                ]
                for experiment, group in subset.groupby("experiment", sort=False):
                    group = group.sort_values("step")
                    ax.plot(group["step"], group[metric], marker="o", label=experiment, color=colors.get(experiment))
                if metric == "bp_pred_mean" and not subset.empty:
                    star = float(subset["bp_star_mean"].iloc[0])
                    ax.axhline(star, color="black", linestyle="--", linewidth=1.0, label="teacher mean")
                ax.set_xlabel("Optimization step")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{branch} current eta={eta_value}: {metric}")
                ax.legend()
                ax.grid(alpha=0.2)
                fig.tight_layout()
                fig.savefig(output_dir / f"{branch}_eta{eta_value}_{metric}.png", dpi=160)
                plt.close(fig)


def comparison_frame(finals: list[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    fields = [
        "experiment",
        "requested_eta1_train_share",
        "realized_eta1_train_share",
        "bp_head_parameter_max_change",
        "non_bp_parameter_max_change",
        "teacher_cache_hash",
    ]
    for branch in ("bp0", "bpi"):
        for eta_value in (0, 1):
            for metric in ("bp_pred_mean", "bp_star_mean", "mae", "mean_regret"):
                fields.append(f"{branch}_eta{eta_value}_{metric}")
    for final in finals:
        rows.append({field: final.get(field) for field in fields})
    return pd.DataFrame(rows)


def verdict_from_results(comparison: pd.DataFrame, eta1_count: int) -> tuple[str, str]:
    if eta1_count == 0 or (comparison["bp_head_parameter_max_change"] <= 0).any():
        return "D", "Training-path bug or missing eta=1 supervision: the controlled experiment is not informative."
    if not {"A_natural", "C_eta50"}.issubset(set(comparison["experiment"])):
        return "NA", "Run both natural and 50/50 experiments before assigning verdict A/B/C."
    natural = comparison.loc[comparison["experiment"] == "A_natural"].iloc[0]
    balanced = comparison.loc[comparison["experiment"] == "C_eta50"].iloc[0]
    eta1_improvements = []
    eta0_degradations = []
    for branch in ("bp0", "bpi"):
        base = float(natural[f"{branch}_eta1_mae"])
        eta1_improvements.append((base - float(balanced[f"{branch}_eta1_mae"])) / max(base, 1e-12))
        eta0_degradations.append(float(balanced[f"{branch}_eta0_mae"]) - float(natural[f"{branch}_eta0_mae"]))
    mean_improvement = float(np.mean(eta1_improvements))
    max_eta0_degradation = float(np.max(eta0_degradations))
    if mean_improvement >= 0.50 and max_eta0_degradation <= 0.05:
        return "A", "Strong support: 50/50 sampling materially repairs eta=1 fit without a large eta=0 MAE cost."
    if mean_improvement >= 0.10:
        return "B", "Partial support: eta balancing improves eta=1 fit, but does not fully resolve the collapse."
    return "C", "Not supported: 50/50 current-eta sampling does not materially improve eta=1 BP fit."


def write_report(
    output_dir: Path,
    *,
    counts: Mapping[str, Any],
    loss_audit: Mapping[str, Any],
    gradient_audit: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
    verdict: str,
    verdict_text: str,
    checkpoint: Path,
    firm_data: Path,
) -> None:
    lines = [
        "# BP current-eta imbalance diagnostic",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- firm data: `{firm_data}`",
        f"- eta0 count/share: {counts['eta0_count']} / {counts['eta0_share']:.6f}",
        f"- eta1 count/share: {counts['eta1_count']} / {counts['eta1_share']:.6f}",
        f"- eta0-to-eta1 count ratio: {counts['eta0_to_eta1_count_ratio']:.6f}",
        "- current eta sampler only: future eta exact integration and ZETA are unchanged",
        "- teacher cache/model, value modules, shared trunk, SDF/FC1, and economic equations are fixed",
        "",
        "## Natural loss contribution",
        "",
        "```json",
        json.dumps(loss_audit, indent=2, sort_keys=True, allow_nan=True),
        "```",
        "",
        "## Conditional gradient contribution",
        "",
        "```json",
        json.dumps(gradient_audit, indent=2, sort_keys=True, allow_nan=True),
        "```",
        "",
        "## Eta switch sensitivity",
        "",
        "```json",
        json.dumps(sensitivity, indent=2, sort_keys=True, allow_nan=True),
        "```",
        "",
        f"## Verdict {verdict}",
        "",
        verdict_text,
        "",
        "Verdict thresholds are diagnostic conventions: A requires at least 50% mean eta1 MAE improvement and no more than 0.05 absolute eta0 MAE deterioration; B requires at least 10% improvement.",
    ]
    (output_dir / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.freeze_shared_trunk:
        raise ValueError("This isolated diagnostic requires --freeze-shared-trunk")
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch-size must be positive")
    experiments = resolve_experiments(args.eta1_train_share)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    loaded = load_analysis_checkpoint(
        args.checkpoint,
        device=device,
        allow_current_config=bool(args.allow_current_config),
        m_source="none",
    )
    base_model = loaded.models["policy_value"]
    teacher_model = loaded.models.get("firm_target")
    teacher_source = "checkpoint_firm_target"
    if teacher_model is None:
        teacher_model = copy.deepcopy(base_model)
        teacher_source = "frozen_policy_value_fallback"
    base_model.eval()
    teacher_model.eval()
    base_model_hash = state_dict_sha256(base_model)
    teacher_hash = state_dict_sha256(teacher_model)
    initial_head_hash = state_dict_sha256(base_model, ("bp0_head.", "bpi_head."))

    with checkpoint_economic_config(loaded.economic_config):
        zeta_before = float(Config.ZETA)
        exact_eta_before = bool(getattr(loaded.hyperparams, "pv_exact_eta_integration_enabled", True))
        episode = make_episode(base_model, teacher_model, loaded.hyperparams, device, args.episode)
        frame = pd.read_pickle(args.firm_data)
        batches = episode._create_firm_batches_from_df(
            frame,
            batch_size=int(args.cache_batch_size),
            n_branches=2,
            eta_resample=False,
        )
        batches = truncate_batches(batches, int(args.max_cache_rows))
        if not batches:
            raise RuntimeError("No matched parent-child firm batches were constructed")
        fixed_cache_list = episode._build_bp_target_cache(batches, teacher_model)
        fixed_cache = flatten_bp_target_cache(fixed_cache_list)
        teacher_cache_hash = bp_teacher_cache_sha256(fixed_cache)
        counts = eta_count_summary(fixed_cache)
        if counts["eta0_count"] == 0 or counts["eta1_count"] == 0:
            raise RuntimeError("The diagnostic cache must contain both current eta strata")

        initial_regrets = evaluate_exact_regrets(episode, batches, teacher_model, base_model)
        cache_audit = pd.DataFrame(
            audit_bp_cache_by_current_eta(base_model, fixed_cache, regrets=initial_regrets)
        )
        cache_audit.to_csv(output_dir / "cache_current_eta_audit.csv", index=False)
        direct_loss_audit = loss_contribution_audit(base_model, fixed_cache, loaded.hyperparams)
        direct_gradient_audit = gradient_contribution_audit(base_model, fixed_cache, loaded.hyperparams)
        total_objective_audit = training_equivalent_total_objective_audit(
            base_model,
            fixed_cache,
            episode._compute_bp_cache_loss,
        )
        loss_audit = {
            "training_equivalent_total": total_objective_audit,
            "direct_branch_only_reference": direct_loss_audit,
        }
        gradient_audit = {
            "training_equivalent_total": total_objective_audit,
            "direct_branch_only_reference": direct_gradient_audit,
        }
        (output_dir / "loss_contribution_audit.json").write_text(
            json.dumps(loss_audit, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8"
        )
        (output_dir / "gradient_contribution_audit.json").write_text(
            json.dumps(gradient_audit, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8"
        )

        sensitivity: Dict[str, Any] = {
            "before": eta_switch_panel_sensitivity(base_model, fixed_cache["parent"]),
        }
        all_metrics = []
        finals = []
        final_models: Dict[str, torch.nn.Module] = {}
        for name, eta1_share in experiments:
            metrics, final, final_model = run_experiment(
                name=name,
                eta1_share=eta1_share,
                base_model=base_model,
                base_episode=episode,
                teacher_model=teacher_model,
                batches=batches,
                cache=fixed_cache,
                args=args,
                output_dir=output_dir / name.replace("A_", "").replace("B_", "").replace("C_", ""),
                initial_head_hash=initial_head_hash,
                teacher_cache_hash=teacher_cache_hash,
            )
            all_metrics.append(metrics)
            finals.append(final)
            final_models[name] = final_model
        if "A_natural" in final_models:
            sensitivity["after_natural"] = eta_switch_panel_sensitivity(
                final_models["A_natural"], fixed_cache["parent"]
            )
        if "C_eta50" in final_models:
            sensitivity["after_eta50"] = eta_switch_panel_sensitivity(
                final_models["C_eta50"], fixed_cache["parent"]
            )
        (output_dir / "eta_switch_sensitivity.json").write_text(
            json.dumps(sensitivity, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8"
        )
        metrics_frame = pd.concat(all_metrics, ignore_index=True)
        comparison = comparison_frame(finals)
        comparison.to_csv(output_dir / "comparison.csv", index=False)
        plot_comparisons(metrics_frame, output_dir)
        verdict, verdict_text = verdict_from_results(comparison, int(counts["eta1_count"]))
        write_report(
            output_dir,
            counts=counts,
            loss_audit=loss_audit,
            gradient_audit=gradient_audit,
            sensitivity=sensitivity,
            verdict=verdict,
            verdict_text=verdict_text,
            checkpoint=args.checkpoint.resolve(),
            firm_data=args.firm_data.resolve(),
        )
        zeta_after = float(Config.ZETA)
        exact_eta_after = bool(getattr(loaded.hyperparams, "pv_exact_eta_integration_enabled", True))
        teacher_cache_hash_after = bp_teacher_cache_sha256(fixed_cache)

    if state_dict_sha256(base_model) != base_model_hash:
        raise RuntimeError("The source policy_value checkpoint model was mutated")
    if state_dict_sha256(teacher_model) != teacher_hash:
        raise RuntimeError("The fixed teacher model was mutated")
    if zeta_after != zeta_before or exact_eta_after != exact_eta_before:
        raise RuntimeError("Diagnostic changed future-eta economic integration settings")
    if teacher_cache_hash_after != teacher_cache_hash:
        raise RuntimeError("Controlled experiments mutated the fixed BP teacher cache")

    metadata = {
        "episode": int(args.episode),
        "checkpoint": str(args.checkpoint.resolve()),
        "firm_data": str(args.firm_data.resolve()),
        "checkpoint_metadata": loaded.metadata,
        "teacher_source": teacher_source,
        "teacher_hash": teacher_hash,
        "teacher_cache_hash": teacher_cache_hash,
        "teacher_cache_hash_after": teacher_cache_hash_after,
        "initial_policy_hash": base_model_hash,
        "initial_bp_head_hash": initial_head_hash,
        "cache_rows": int(fixed_cache["parent"].shape[0]),
        "current_eta_counts": counts,
        "formal_bp_cache_parent_eta_source": "parent_state[:, 2]",
        "formal_bp_cache_sampling": "natural cache batches; no current-eta grouping",
        "formal_bp_loss_reduction": "confidence/sample-weighted element loss followed by plain row mean",
        "formal_validation_distribution": "natural fixed holdout cache",
        "legacy_eta_resampling_scope": "future eta_next_active only; not used here",
        "future_eta_zeta_before": zeta_before,
        "future_eta_zeta_after": zeta_after,
        "future_eta_exact_enabled_before": exact_eta_before,
        "future_eta_exact_enabled_after": exact_eta_after,
        "experiments": [name for name, _ in experiments],
        "verdict": verdict,
        "verdict_text": verdict_text,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8"
    )
    print(f"BP current-eta diagnostic complete: {output_dir}")
    print(f"eta0/eta1 counts: {counts['eta0_count']}/{counts['eta1_count']}")
    print(f"verdict: {verdict} - {verdict_text}")


if __name__ == "__main__":
    main()
