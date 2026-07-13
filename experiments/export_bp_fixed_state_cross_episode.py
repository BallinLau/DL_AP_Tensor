from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from experiments.export_bp_deep_diagnostics import make_episode_batches, stable_logit_with_censoring  # noqa: E402
from experiments.run_utils import build_hyperparams, build_models  # noqa: E402
from losses import P0Loss, PILoss  # noqa: E402
from training.bp_grid_teacher import BPGridTeacher  # noqa: E402


def parse_episodes(value: str) -> List[int]:
    return [int(x) for x in value.replace(",", " ").split() if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-state cross-episode BP probe.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--episodes", type=parse_episodes, default=parse_episodes("0 1 2"))
    parser.add_argument("--reference-episode", type=int, default=2)
    parser.add_argument("--reference-firm-pkl", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--states-per-group", type=int, default=20)
    parser.add_argument("--flip-threshold", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-branches", type=int, default=2)
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def choose_bucket_indices(values: torch.Tensor, n_each: int) -> Dict[str, torch.Tensor]:
    order = torch.argsort(values.reshape(-1))
    n = order.numel()
    n_each = max(1, min(int(n_each), n))
    low = order[:n_each]
    high = order[-n_each:]
    mid_center = n // 2
    mid_start = max(0, mid_center - n_each // 2)
    mid_stop = min(n, mid_start + n_each)
    mid = order[mid_start:mid_stop]
    return {"low": low, "middle": mid, "high": high}


def build_fixed_probe_panel(
    tensors: Dict[str, torch.Tensor],
    states_per_group: int,
) -> Dict[str, torch.Tensor | pd.DataFrame]:
    parent = tensors["parent"][:, :7]
    active = parent[:, 2] > 0.5
    active_idx = torch.where(active)[0]
    if active_idx.numel() == 0:
        raise RuntimeError("No active refinancing states found in reference firm data.")

    active_parent = parent[active_idx]
    rows = []
    selected = []
    seen = set()
    for axis_name, col in [("z", 1), ("b", 0)]:
        buckets = choose_bucket_indices(active_parent[:, col], states_per_group)
        for bucket, local_idx in buckets.items():
            for idx in local_idx.tolist():
                global_idx = int(active_idx[idx].item())
                source = int(tensors["source_index"][global_idx].item())
                if source in seen:
                    continue
                seen.add(source)
                selected.append(global_idx)
                rows.append(
                    {
                        "panel_pos": len(selected) - 1,
                        "source_index": source,
                        "probe_group": f"{bucket}-{axis_name}",
                        "selection_axis": axis_name,
                        "b": float(parent[global_idx, 0].item()),
                        "z": float(parent[global_idx, 1].item()),
                        "eta": float(parent[global_idx, 2].item()),
                        "i": float(parent[global_idx, 3].item()),
                        "x": float(parent[global_idx, 4].item()),
                        "hatcf": float(parent[global_idx, 5].item()),
                        "lnkf": float(parent[global_idx, 6].item()),
                    }
                )
    idx_tensor = torch.tensor(selected, device=parent.device, dtype=torch.long)
    return {
        "parent": parent[idx_tensor],
        "children": [tensors["child0"][idx_tensor, :7], tensors["child1"][idx_tensor, :7]],
        "m_list": [tensors["child0"][idx_tensor, 7:8], tensors["child1"][idx_tensor, 7:8]],
        "source_index": tensors["source_index"][idx_tensor],
        "panel": pd.DataFrame(rows),
    }


def compute_episode_rows(
    *,
    run_root: Path,
    episode: int,
    panel: pd.DataFrame,
    parent: torch.Tensor,
    children: List[torch.Tensor],
    m_list: List[torch.Tensor],
    hp,
    device: torch.device,
) -> List[Dict[str, object]]:
    ckpt_dir = run_root / "checkpoints"
    require_file(ckpt_dir / f"ep{episode}_policy_value.pt", f"EP{episode} policy/value checkpoint")
    models = build_models(device=device, ckpt_dir=ckpt_dir, ckpt_prefix=f"ep{episode}", strict=True)
    model = models["policy_value"].eval()
    target_model = copy.deepcopy(model).to(device).eval()
    for p in target_model.parameters():
        p.requires_grad_(False)
    teacher = BPGridTeacher.from_hyperparams(target_model, P0Loss(), PILoss(), hp)
    with torch.no_grad():
        out = model(parent)
        mix_weight = out.bar_i_cond.clamp(0.0, 1.0)
        bp_mix = (1.0 - mix_weight) * out.bp0 + mix_weight * out.bpI
        bp0_logit, _, _ = stable_logit_with_censoring(out.bp0)
        bpI_logit, _, _ = stable_logit_with_censoring(out.bpI)
        grids = {
            "p0": teacher.compute(parent, children, m_list, branch="p0", bp_pred=out.bp0),
            "pi": teacher.compute(parent, children, m_list, branch="pi", bp_pred=out.bpI),
            "mix": teacher.compute(parent, children, m_list, branch="mix", bp_pred=bp_mix, mix_weight=mix_weight),
        }

    rows = []
    for i, panel_row in panel.reset_index(drop=True).iterrows():
        for branch, pred, teacher_key in [
            ("p0", out.bp0, "bp0"),
            ("pi", out.bpI, "bpI"),
            ("mix", bp_mix, "bp_mix"),
        ]:
            teacher_bp = grids[branch]["bp_star"][i]
            rows.append(
                {
                    "episode": episode,
                    "source_index": int(panel_row["source_index"]),
                    "probe_group": panel_row["probe_group"],
                    "branch": branch,
                    "teacher_semantics": "checkpoint_online_greedy_proxy",
                    "b": float(parent[i, 0].item()),
                    "z": float(parent[i, 1].item()),
                    "eta": float(parent[i, 2].item()),
                    "i": float(parent[i, 3].item()),
                    "x": float(parent[i, 4].item()),
                    "hatcf": float(parent[i, 5].item()),
                    "lnkf": float(parent[i, 6].item()),
                    "bp0_pred": float(out.bp0[i].item()),
                    "bpI_pred": float(out.bpI[i].item()),
                    "bp_mix_pred": float(bp_mix[i].item()),
                    "bp0_teacher": float(grids["p0"]["bp_star"][i].item()),
                    "bpI_teacher": float(grids["pi"]["bp_star"][i].item()),
                    "bp_mix_teacher": float(grids["mix"]["bp_star"][i].item()),
                    "bp0_gap": float((grids["p0"]["bp_star"][i] - out.bp0[i]).item()),
                    "bpI_gap": float((grids["pi"]["bp_star"][i] - out.bpI[i]).item()),
                    "bp_mix_gap": float((grids["mix"]["bp_star"][i] - bp_mix[i]).item()),
                    "bp_pred": float(pred[i].item()),
                    "bp_teacher": float(teacher_bp.item()),
                    "bp_gap": float((teacher_bp - pred[i]).item()),
                    "regret": float(grids[branch]["regret"][i].item()),
                    "logit_bp0": float(bp0_logit[i].item()),
                    "logit_bpI": float(bpI_logit[i].item()),
                }
            )
    return rows


def attach_flip_flags(long_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    long_df = long_df.copy()
    long_df["teacher_flip"] = False
    long_df["policy_flip"] = False
    long_df["policy_flip_without_teacher_flip"] = False
    long_df["teacher_flip_not_followed_by_policy"] = False
    episodes = sorted(long_df["episode"].unique())
    if len(episodes) < 2:
        return long_df
    prev_ep, last_ep = episodes[-2], episodes[-1]
    key = ["source_index", "branch"]
    prev = long_df[long_df["episode"] == prev_ep][key + ["bp_pred", "bp_teacher"]]
    last = long_df[long_df["episode"] == last_ep][key + ["bp_pred", "bp_teacher"]]
    merged = last.merge(prev, on=key, suffixes=("_last", "_prev"))
    merged["teacher_flip"] = (merged["bp_teacher_last"] - merged["bp_teacher_prev"]).abs() > threshold
    merged["policy_flip"] = (merged["bp_pred_last"] - merged["bp_pred_prev"]).abs() > threshold
    merged["policy_flip_without_teacher_flip"] = merged["policy_flip"] & ~merged["teacher_flip"]
    merged["teacher_flip_not_followed_by_policy"] = merged["teacher_flip"] & ~merged["policy_flip"]
    flag_cols = [
        "teacher_flip",
        "policy_flip",
        "policy_flip_without_teacher_flip",
        "teacher_flip_not_followed_by_policy",
    ]
    for _, row in merged.iterrows():
        mask = (
            (long_df["episode"] == last_ep)
            & (long_df["source_index"] == row["source_index"])
            & (long_df["branch"] == row["branch"])
        )
        for col in flag_cols:
            long_df.loc[mask, col] = bool(row[col])
    return long_df


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    Config.DEVICE = device
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_firm = require_file(args.reference_firm_pkl, "reference firm pickle")
    hp = build_hyperparams()
    hp.pv_eta_resample_enabled = False
    hp.max_firm_train_units = 0

    ref_models = build_models(
        device=device,
        ckpt_dir=run_root / "checkpoints",
        ckpt_prefix=f"ep{args.reference_episode}",
        strict=True,
    )
    tensors = make_episode_batches(
        reference_firm,
        ref_models["policy_value"],
        hp,
        device,
        args.batch_size,
        args.n_branches,
    )
    fixed = build_fixed_probe_panel(tensors, args.states_per_group)
    panel = fixed["panel"]
    panel.to_csv(output_dir / "fixed_probe_states.csv", index=False)
    torch.save(fixed["parent"].detach().cpu(), output_dir / "fixed_probe_parent.pt")
    torch.save([c.detach().cpu() for c in fixed["children"]], output_dir / "fixed_probe_children.pt")
    torch.save([m.detach().cpu() for m in fixed["m_list"]], output_dir / "fixed_probe_m.pt")

    rows: List[Dict[str, object]] = []
    for ep in args.episodes:
        rows.extend(
            compute_episode_rows(
                run_root=run_root,
                episode=ep,
                panel=panel,
                parent=fixed["parent"],
                children=fixed["children"],
                m_list=fixed["m_list"],
                hp=hp,
                device=device,
            )
        )
    long_df = attach_flip_flags(pd.DataFrame(rows), args.flip_threshold)
    if long_df.duplicated(["episode", "source_index", "branch"]).any():
        raise RuntimeError("Duplicate episode/source_index/branch rows in fixed-state export.")

    summary = (
        long_df.groupby(["episode", "probe_group", "branch"])
        .agg(
            n_states=("source_index", "nunique"),
            bp_pred_mean=("bp_pred", "mean"),
            bp_pred_p10=("bp_pred", lambda x: x.quantile(0.10)),
            bp_pred_p50=("bp_pred", "median"),
            bp_pred_p90=("bp_pred", lambda x: x.quantile(0.90)),
            bp_teacher_mean=("bp_teacher", "mean"),
            bp_teacher_p10=("bp_teacher", lambda x: x.quantile(0.10)),
            bp_teacher_p50=("bp_teacher", "median"),
            bp_teacher_p90=("bp_teacher", lambda x: x.quantile(0.90)),
            bp_gap_mean=("bp_gap", "mean"),
            regret_mean=("regret", "mean"),
            teacher_flip_share=("teacher_flip", "mean"),
            policy_flip_share=("policy_flip", "mean"),
            policy_flip_without_teacher_flip_share=("policy_flip_without_teacher_flip", "mean"),
            teacher_flip_not_followed_by_policy_share=("teacher_flip_not_followed_by_policy", "mean"),
        )
        .reset_index()
    )

    long_path = output_dir / "fixed_state_cross_episode_long.csv"
    summary_path = output_dir / "fixed_state_cross_episode_summary.csv"
    long_df.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)
    print("Saved:")
    print(long_path)
    print(summary_path)
    print(panel.head().to_string(index=False))


if __name__ == "__main__":
    main()
