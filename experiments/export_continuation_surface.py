from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from experiments.export_bp_deep_diagnostics import make_episode_batches  # noqa: E402
from experiments.run_utils import build_hyperparams, build_models  # noqa: E402
from losses.q_loss import compute_q_survival_recovery_components  # noqa: E402


def validate_q_decomposition_frame(q_long: pd.DataFrame, *, tolerance: float = 1e-7) -> None:
    required = [
        "q_issue",
        "q_target_total",
        "q_training_residual",
        "q_issue_minus_target",
        "bar_i",
        "bar_i_multiplier",
        "bar_i_mode",
    ]
    missing = [col for col in required if col not in q_long.columns]
    if missing:
        raise RuntimeError(f"Q decomposition CSV is missing required columns: {missing}")

    numeric_cols = [
        "q_issue",
        "q_target_total",
        "q_training_residual",
        "q_issue_minus_target",
        "bar_i",
        "bar_i_multiplier",
    ]
    values = q_long[numeric_cols].apply(pd.to_numeric, errors="coerce")
    finite = torch.isfinite(torch.as_tensor(values.to_numpy(dtype="float64")))
    if not bool(finite.all().item()):
        raise RuntimeError("Q decomposition residual columns contain non-finite values.")

    training_identity = (
        values["q_training_residual"]
        - (values["q_target_total"] - values["q_issue"])
    ).abs().max()
    sign_identity = (
        values["q_issue_minus_target"] + values["q_training_residual"]
    ).abs().max()
    max_error = max(float(training_identity), float(sign_identity))
    if max_error > tolerance:
        raise RuntimeError(
            "Q decomposition residual sign identity failed: "
            f"max_error={max_error:.6g}, tolerance={tolerance:.6g}"
        )


def _extract_bar_i(model_output) -> torch.Tensor:
    if isinstance(model_output, dict):
        return model_output["bar_i"]
    return model_output.bar_i


def compute_issue_bar_i(
    model,
    parent: torch.Tensor,
    issue_states: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    if mode == "fixed_parent":
        return _extract_bar_i(model(parent)).expand(issue_states.shape[0], -1)
    if mode == "recompute_issue_state":
        return _extract_bar_i(model(issue_states))
    raise ValueError(f"Unknown bar_i mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export continuation and Q decomposition surfaces.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--base-state-source-index", type=int, required=True)
    parser.add_argument("--firm-pkl", type=Path, required=True)
    parser.add_argument("--b-grid-size", type=int, default=21)
    parser.add_argument("--z-min", type=float, default=-2.0)
    parser.add_argument("--z-max", type=float, default=2.0)
    parser.add_argument("--z-grid-size", type=int, default=41)
    parser.add_argument("--m-mode", choices=["observed_child_mean", "fixed"], default="observed_child_mean")
    parser.add_argument("--m-fixed", type=float, default=1.0)
    parser.add_argument(
        "--bar-i-mode",
        choices=["fixed_parent", "recompute_issue_state"],
        default="fixed_parent",
        help=(
            "fixed_parent keeps bar_i(b)=bar_i(base_parent) as a partial-equilibrium "
            "debt slice; recompute_issue_state evaluates bar_i on each issue_state(b)."
        ),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-branches", type=int, default=2)
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def select_base(
    tensors: Dict[str, torch.Tensor],
    source_index: int,
) -> tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    src = tensors["source_index"].detach().cpu().to(torch.long)
    matches = torch.where(src == int(source_index))[0]
    if matches.numel() == 0:
        raise RuntimeError(f"source_index={source_index} not found in firm data.")
    idx = matches[0].to(device=tensors["parent"].device)
    parent = tensors["parent"][idx:idx + 1, :7]
    children = [tensors["child0"][idx:idx + 1, :7], tensors["child1"][idx:idx + 1, :7]]
    m = torch.stack([tensors["child0"][idx, 7], tensors["child1"][idx, 7]]).mean().reshape(1, 1)
    return parent, children, m


def make_surface_states(
    child_base: torch.Tensor,
    b_grid_size: int,
    z_min: float,
    z_max: float,
    z_grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b_values = torch.linspace(0.0, 1.0, steps=b_grid_size, device=child_base.device, dtype=child_base.dtype)
    z_values = torch.linspace(z_min, z_max, steps=z_grid_size, device=child_base.device, dtype=child_base.dtype)
    mesh_b, mesh_z = torch.meshgrid(b_values, z_values, indexing="ij")
    states = child_base.expand(mesh_b.numel(), -1).clone()
    states[:, 0:1] = mesh_b.reshape(-1, 1)
    states[:, 1:2] = mesh_z.reshape(-1, 1)
    return states, mesh_b.reshape(-1, 1), mesh_z.reshape(-1, 1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    Config.DEVICE = device
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    firm_pkl = require_file(args.firm_pkl, "firm pickle")
    ckpt_dir = run_root / "checkpoints"
    require_file(ckpt_dir / f"ep{args.episode}_policy_value.pt", "policy/value checkpoint")
    hp = build_hyperparams()
    hp.pv_eta_resample_enabled = False
    hp.max_firm_train_units = 0
    models = build_models(device=device, ckpt_dir=ckpt_dir, ckpt_prefix=f"ep{args.episode}", strict=True)
    model = models["policy_value"].eval()
    target_model = copy.deepcopy(model).to(device).eval()
    for p in target_model.parameters():
        p.requires_grad_(False)

    tensors = make_episode_batches(firm_pkl, model, hp, device, args.batch_size, args.n_branches)
    parent, children, observed_m = select_base(tensors, args.base_state_source_index)
    child_base = children[0]
    surface_states, b_candidate, z_child = make_surface_states(
        child_base,
        args.b_grid_size,
        args.z_min,
        args.z_max,
        args.z_grid_size,
    )
    if args.m_mode == "fixed":
        m_used = torch.full_like(b_candidate, float(args.m_fixed))
    else:
        m_used = observed_m.expand_as(b_candidate)

    with torch.no_grad():
        equity = target_model.forward_equity(surface_states)
        phat = equity["Phat"]
        p_child = equity["P"]
        survival = equity["survival_prob"]
        default = equity["bar_z"].clamp(0.0, 1.0)

    rows = []
    for branch, multiplier in [("p0", 1.0), ("pi", Config.G)]:
        continuation = multiplier * m_used * p_child
        for i in range(surface_states.shape[0]):
            rows.append(
                {
                    "episode": args.episode,
                    "source_index": args.base_state_source_index,
                    "branch": branch,
                    "b_candidate": float(b_candidate[i].item()),
                    "z_child": float(z_child[i].item()),
                    "phat_child": float(phat[i].item()),
                    "p_child": float(p_child[i].item()),
                    "survival_child": float(survival[i].item()),
                    "default_child": float(default[i].item()),
                    "m_used": float(m_used[i].item()),
                    "continuation_p0": float((m_used[i] * p_child[i]).item()),
                    "continuation_pi": float((Config.G * m_used[i] * p_child[i]).item()),
                }
            )

    q_rows = []
    b_values = torch.linspace(0.0, 1.0, steps=args.b_grid_size, device=device, dtype=parent.dtype).reshape(-1, 1)
    with torch.no_grad():
        issue_states = parent.expand(b_values.shape[0], -1).clone()
        issue_states[:, 0:1] = b_values
        q_issue = target_model._q_output(issue_states)
        bar_i_issue = compute_issue_bar_i(
            target_model,
            parent,
            issue_states,
            mode=args.bar_i_mode,
        ).clamp(0.0, 1.0)
        multiplier = bar_i_issue * (Config.G - 1.0) + 1.0
        q_target_survival_sum = torch.zeros_like(b_values)
        q_target_recovery_sum = torch.zeros_like(b_values)
        q_target_total_sum = torch.zeros_like(b_values)
        q_training_residual_sum = torch.zeros_like(b_values)
        q_issue_minus_target_sum = torch.zeros_like(b_values)
        default_sum = torch.zeros_like(b_values)
        m_sum = torch.zeros_like(b_values)
        for child_idx, child in enumerate(children):
            child_state = child.expand(b_values.shape[0], -1).clone()
            b_sp = b_values / multiplier.clamp_min(1e-6)
            child_state[:, 0:1] = b_sp
            child_q = target_model._q_output(child_state)
            child_equity = target_model.forward_equity(child_state)
            child_default = child_equity["bar_z"].clamp(0.0, 1.0)
            child_m = (
                torch.full_like(b_values, float(args.m_fixed))
                if args.m_mode == "fixed"
                else tensors[f"child{child_idx}"][torch.where(tensors["source_index"].detach().cpu().to(torch.long) == int(args.base_state_source_index))[0][0], 7].to(device=device, dtype=parent.dtype).reshape(1, 1).expand_as(b_values)
            )
            components = compute_q_survival_recovery_components(
                Q=q_issue,
                b=b_values,
                bar_i=bar_i_issue,
                M=child_m,
                Qsp=child_q,
                bar_z=child_default,
                x_child=child_state[:, 4:5],
                z_child=child_state[:, 1:2],
                g=Config.G,
                delta=Config.DELTA,
                phi=Config.PHI,
            )
            q_target_survival_sum = q_target_survival_sum + components["q_target_survival"]
            q_target_recovery_sum = q_target_recovery_sum + components["q_target_recovery"]
            q_target_total_sum = q_target_total_sum + components["q_target_total"]
            q_training_residual_sum = q_training_residual_sum + components["q_training_residual"]
            q_issue_minus_target_sum = q_issue_minus_target_sum + components["q_issue_minus_target"]
            default_sum = default_sum + child_default
            m_sum = m_sum + child_m
        n_children = float(len(children))
        q_target_survival = q_target_survival_sum / n_children
        q_target_recovery = q_target_recovery_sum / n_children
        q_target_total = q_target_total_sum / n_children
        q_training_residual = q_training_residual_sum / n_children
        q_issue_minus_target = q_issue_minus_target_sum / n_children
        recovery_share = q_target_recovery / q_target_total.clamp_min(1e-12)
        default_mean = default_sum / n_children
        m_mean = m_sum / n_children
        for i in range(b_values.shape[0]):
            q_rows.append(
                {
                    "episode": args.episode,
                    "source_index": args.base_state_source_index,
                    "branch": "formal_q",
                    "b_candidate": float(b_values[i].item()),
                    "q_issue": float(q_issue[i].item()),
                    "q_target_survival": float(q_target_survival[i].item()),
                    "q_target_recovery": float(q_target_recovery[i].item()),
                    "q_target_total": float(q_target_total[i].item()),
                    "q_training_residual": float(q_training_residual[i].item()),
                    "q_issue_minus_target": float(q_issue_minus_target[i].item()),
                    "recovery_share": float(recovery_share[i].item()),
                    "default_child_mean": float(default_mean[i].item()),
                    "m_used_mean": float(m_mean[i].item()),
                    "bar_i": float(bar_i_issue[i].item()),
                    "bar_i_multiplier": float(multiplier[i].item()),
                    "bar_i_mode": args.bar_i_mode,
                }
            )

    surface = pd.DataFrame(rows)
    q_long = pd.DataFrame(q_rows)
    validate_q_decomposition_frame(q_long)
    boundary_rows = []
    for b_val, group in surface[surface["branch"] == "p0"].groupby("b_candidate"):
        survived = group[group["survival_child"] >= 0.5].sort_values("z_child")
        boundary_found = not survived.empty
        boundary_rows.append(
            {
                "episode": args.episode,
                "source_index": args.base_state_source_index,
                "b_candidate": b_val,
                "z_survival_boundary": float(survived["z_child"].iloc[0]) if boundary_found else float(args.z_max),
                "boundary_found": boundary_found,
            }
        )
    boundary = pd.DataFrame(boundary_rows)
    z_vals = boundary["z_survival_boundary"].dropna().to_numpy()
    violation_count = int(((z_vals[1:] - z_vals[:-1]) < -1e-8).sum()) if len(z_vals) > 1 else 0
    boundary["boundary_monotonicity_violation_count"] = violation_count
    q_summary = (
        q_long.groupby(["episode", "branch", "bar_i_mode"])
        .agg(
            n_rows=("source_index", "size"),
            q_issue_mean=("q_issue", "mean"),
            q_target_survival_mean=("q_target_survival", "mean"),
            q_target_recovery_mean=("q_target_recovery", "mean"),
            q_target_total_mean=("q_target_total", "mean"),
            q_training_residual_mean=("q_training_residual", "mean"),
            q_issue_minus_target_mean=("q_issue_minus_target", "mean"),
            recovery_share_mean=("recovery_share", "mean"),
        )
        .reset_index()
    )

    surface_path = output_dir / f"ep{args.episode}_continuation_surface.csv"
    boundary_path = output_dir / f"ep{args.episode}_default_boundary.csv"
    q_long_path = output_dir / f"ep{args.episode}_bond_value_decomposition_long.csv"
    q_summary_path = output_dir / f"ep{args.episode}_bond_value_decomposition_summary.csv"
    surface.to_csv(surface_path, index=False)
    boundary.to_csv(boundary_path, index=False)
    q_long.to_csv(q_long_path, index=False)
    q_summary.to_csv(q_summary_path, index=False)
    print("Saved:")
    print(surface_path)
    print(boundary_path)
    print(q_long_path)
    print(q_summary_path)
    print("bar_i_mode:", args.bar_i_mode)
    print("boundary_monotonicity_violation_count:", violation_count)


if __name__ == "__main__":
    main()
