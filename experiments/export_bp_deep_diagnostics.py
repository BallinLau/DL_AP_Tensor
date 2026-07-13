from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from experiments.run_utils import build_hyperparams, build_models  # noqa: E402
from losses.utils import compute_cashflow  # noqa: E402
from training.episode import Episode  # noqa: E402
from utils.firm_transition import apply_refinancing_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export deep BP diagnostics from an existing target-grid "
            "decomposition export."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--firm-pkl", type=Path, required=True)
    parser.add_argument("--decomposition-long", type=Path, required=True)
    parser.add_argument("--decomposition-summary", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-branches", type=int, default=2)
    return parser.parse_args()


def cat_batches(batches: Iterable[Dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    values = [batch[key] for batch in batches if batch.get(key) is not None]
    if not values:
        raise RuntimeError(f"No values found for batch key: {key}")
    return torch.cat(values, dim=0)


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def make_episode_batches(
    firm_pkl: Path,
    model,
    hp,
    device: torch.device,
    batch_size: int,
    n_branches: int,
) -> Dict[str, torch.Tensor]:
    episode = Episode.__new__(Episode)
    episode.device = device
    episode.hyperparams = hp
    episode.models = {"policy_value": model}

    df = pd.read_pickle(firm_pkl)
    batches = episode._create_firm_batches_from_df(
        df,
        batch_size=batch_size,
        n_branches=n_branches,
        eta_resample=False,
    )
    if not batches:
        raise RuntimeError("No parent-child batches were constructed.")

    tensors = {
        "parent": cat_batches(batches, "parent"),
        "child0": cat_batches(batches, "child0"),
        "child1": cat_batches(batches, "child1"),
        "source_index": cat_batches(batches, "parent_source_index").reshape(-1),
    }
    for name in ["parent", "child0", "child1"]:
        tensor = tensors[name]
        print(f"{name}.shape={tuple(tensor.shape)}")
        if tensor.dim() != 2 or tensor.shape[1] != 8:
            raise RuntimeError(
                f"Expected {name} to have 7 state columns plus M, got {tuple(tensor.shape)}"
            )
    return tensors


def select_exported_states(
    summary_df: pd.DataFrame,
    tensors: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    state_map = (
        summary_df[["state_pos", "source_index"]]
        .drop_duplicates()
        .sort_values("state_pos")
        .reset_index(drop=True)
    )
    if state_map.empty:
        raise RuntimeError("Decomposition summary has no exported states.")

    source_index = tensors["source_index"].detach().cpu().to(torch.long)
    lookup = {int(src.item()): pos for pos, src in enumerate(source_index)}
    selected_positions = []
    for src in state_map["source_index"].astype(int).tolist():
        if src not in lookup:
            raise RuntimeError(f"source_index={src} from summary not found in firm batches.")
        selected_positions.append(lookup[src])

    idx = torch.tensor(
        selected_positions,
        device=tensors["parent"].device,
        dtype=torch.long,
    )
    return {
        "parent": tensors["parent"][idx, :7],
        "children": [tensors["child0"][idx, :7], tensors["child1"][idx, :7]],
        "m_list": [tensors["child0"][idx, 7:8], tensors["child1"][idx, 7:8]],
        "source_index": tensors["source_index"][idx],
        "state_map": state_map,
    }


def stable_logit(bp: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    bp = bp.clamp(eps, 1.0 - eps)
    return torch.log(bp / (1.0 - bp))


def module_grad_norm(module: torch.nn.Module) -> float:
    sq_sum = 0.0
    for param in module.parameters():
        if param.grad is not None:
            sq_sum += float(param.grad.detach().pow(2).sum().item())
    return float(sq_sum ** 0.5)


def quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.detach().reshape(-1), q).item())


def branch_targets(summary_df: pd.DataFrame, branch: str, device: torch.device) -> torch.Tensor:
    target = (
        summary_df.loc[summary_df["branch"] == branch]
        .sort_values("state_pos")["bp_star_teacher"]
        .to_numpy(dtype=np.float64)
    )
    if target.size == 0:
        raise RuntimeError(f"No summary rows found for branch={branch}")
    return torch.as_tensor(target, device=device, dtype=torch.float32).reshape(-1, 1)


def export_logits_and_gradients(
    *,
    model,
    parent_state: torch.Tensor,
    source_index: torch.Tensor,
    summary_df: pd.DataFrame,
    episode: int,
    output_dir: Path,
) -> None:
    model.train()
    out = model(parent_state)
    mix_weight = out.bar_i_cond.clamp(0.0, 1.0)
    bp_mix = (1.0 - mix_weight) * out.bp0 + mix_weight * out.bpI
    bp0_logit = stable_logit(out.bp0)
    bpI_logit = stable_logit(out.bpI)
    bp0_deriv = out.bp0.clamp(0.0, 1.0) * (1.0 - out.bp0.clamp(0.0, 1.0))
    bpI_deriv = out.bpI.clamp(0.0, 1.0) * (1.0 - out.bpI.clamp(0.0, 1.0))

    state_rows: List[Dict[str, object]] = []
    summary_sorted = summary_df.sort_values(["branch", "state_pos"])
    state_lookup = {int(pos): i for i, pos in enumerate(sorted(summary_df["state_pos"].unique()))}
    for _, row in summary_sorted.iterrows():
        i = state_lookup[int(row["state_pos"])]
        state_rows.append(
            {
                "episode": episode,
                "source_index": int(source_index[i].item()),
                "branch": row["branch"],
                "b": float(parent_state[i, 0].item()),
                "z": float(parent_state[i, 1].item()),
                "eta": float(parent_state[i, 2].item()),
                "i": float(parent_state[i, 3].item()),
                "x": float(parent_state[i, 4].item()),
                "hatcf": float(parent_state[i, 5].item()),
                "lnkf": float(parent_state[i, 6].item()),
                "bp0_pred": float(out.bp0[i].detach().item()),
                "bpI_pred": float(out.bpI[i].detach().item()),
                "bp_mix_pred": float(bp_mix[i].detach().item()),
                "bp_star_teacher": float(row["bp_star_teacher"]),
                "bp0_logit": float(bp0_logit[i].detach().item()),
                "bpI_logit": float(bpI_logit[i].detach().item()),
                "bp0_sigmoid_derivative": float(bp0_deriv[i].detach().item()),
                "bpI_sigmoid_derivative": float(bpI_deriv[i].detach().item()),
            }
        )

    gradient_rows: List[Dict[str, object]] = []
    branches = {
        "p0": out.bp0,
        "pi": out.bpI,
        "mix": bp_mix,
    }
    for branch, pred in branches.items():
        target = branch_targets(summary_df, branch, parent_state.device).to(pred.dtype)
        model.zero_grad(set_to_none=True)
        loss = F.mse_loss(pred, target)
        loss.backward(retain_graph=True)
        gradient_rows.append(
            {
                "episode": episode,
                "branch": branch,
                "n_states": int(parent_state.shape[0]),
                "diagnostic_target_loss": float(loss.detach().item()),
                "bp0_head_grad_norm": module_grad_norm(model.bp0_head),
                "bpI_head_grad_norm": module_grad_norm(model.bpi_head),
                "policy_encoder_grad_norm": module_grad_norm(model.policy_encoder),
                "bp0_logit_mean": float(bp0_logit.detach().mean().item()),
                "bp0_logit_p01": quantile(bp0_logit, 0.01),
                "bp0_logit_p50": quantile(bp0_logit, 0.50),
                "bp0_logit_p99": quantile(bp0_logit, 0.99),
                "bpI_logit_mean": float(bpI_logit.detach().mean().item()),
                "bpI_logit_p01": quantile(bpI_logit, 0.01),
                "bpI_logit_p50": quantile(bpI_logit, 0.50),
                "bpI_logit_p99": quantile(bpI_logit, 0.99),
                "bp0_sigmoid_derivative_mean": float(bp0_deriv.detach().mean().item()),
                "bpI_sigmoid_derivative_mean": float(bpI_deriv.detach().mean().item()),
                "optimizer_steps_observed": "NA",
                "optimizer_steps_source": "unavailable",
            }
        )
    model.zero_grad(set_to_none=True)
    model.eval()

    pd.DataFrame(state_rows).to_csv(output_dir / f"ep{episode}_bp_logits_states.csv", index=False)
    pd.DataFrame(gradient_rows).to_csv(output_dir / f"ep{episode}_bp_gradient_summary.csv", index=False)


def total_recovery_for_rows(
    children: List[torch.Tensor],
    row_state_pos: torch.Tensor,
    effective_b: torch.Tensor,
) -> torch.Tensor:
    recovery_terms = []
    for child in children:
        child_rows = child[row_state_pos]
        x_child = child_rows[:, 4:5]
        z_child = child_rows[:, 1:2]
        recovery_unit = Config.PHI * (1.0 - Config.DELTA + torch.exp(x_child + z_child))
        recovery_terms.append(effective_b * recovery_unit)
    return torch.stack(recovery_terms, dim=1).mean(dim=1)


def export_cashflow_components(
    *,
    model,
    parent_state: torch.Tensor,
    children: List[torch.Tensor],
    long_df: pd.DataFrame,
    episode: int,
    output_dir: Path,
) -> None:
    model.eval()
    with torch.no_grad():
        q_current_all = model._q_output(parent_state).detach()
        out = model(parent_state)
        mix_weight_all = out.bar_i_cond.clamp(0.0, 1.0).detach()

    rows: List[pd.DataFrame] = []
    for branch, group in long_df.groupby("branch", sort=False):
        group = group.copy().reset_index(drop=True)
        state_pos = torch.as_tensor(
            group["state_pos"].to_numpy(dtype=np.int64),
            device=parent_state.device,
            dtype=torch.long,
        )
        bp_candidate = torch.as_tensor(
            group["bp_candidate"].to_numpy(dtype=np.float64),
            device=parent_state.device,
            dtype=parent_state.dtype,
        ).reshape(-1, 1)
        q_issue = torch.as_tensor(
            group["q_issue"].to_numpy(dtype=np.float64),
            device=parent_state.device,
            dtype=parent_state.dtype,
        ).reshape(-1, 1)
        parent_rows = parent_state[state_pos]
        b = parent_rows[:, 0:1]
        z = parent_rows[:, 1:2]
        eta = parent_rows[:, 2:3].clamp(0.0, 1.0)
        invest = parent_rows[:, 3:4]
        x = parent_rows[:, 4:5]
        q_current = q_current_all[state_pos]
        effective_b = apply_refinancing_policy(b, bp_candidate, eta)

        production = compute_cashflow(x, z, b, Config.DELTA, Config.TAU)
        debt_p0 = ((1.0 - Config.KAPPA_B) * q_issue - q_current) * eta
        raw_p0 = production + debt_p0
        equity_p0 = Config.KAPPA_E * F.relu(-raw_p0)
        cf_p0 = raw_p0 - equity_p0

        debt_pi = ((1.0 - Config.KAPPA_B) * Config.G * q_issue - q_current) * eta
        raw_pi = production - invest + debt_pi
        equity_pi = Config.KAPPA_E * F.relu(-raw_pi)
        cf_pi = raw_pi - equity_pi

        if branch == "p0":
            cashflow_reconstructed = cf_p0
            debt_adjustment = debt_p0
            equity_cost = equity_p0
            investment_adjustment = torch.zeros_like(invest)
        elif branch == "pi":
            cashflow_reconstructed = cf_pi
            debt_adjustment = debt_pi
            equity_cost = equity_pi
            investment_adjustment = -invest
        elif branch == "mix":
            mix_weight = mix_weight_all[state_pos]
            cashflow_reconstructed = (1.0 - mix_weight) * cf_p0 + mix_weight * cf_pi
            debt_adjustment = (1.0 - mix_weight) * debt_p0 + mix_weight * debt_pi
            equity_cost = (1.0 - mix_weight) * equity_p0 + mix_weight * equity_pi
            investment_adjustment = -mix_weight * invest
        else:
            raise ValueError(f"Unexpected branch: {branch}")

        exported_cashflow = torch.as_tensor(
            group["cashflow"].to_numpy(dtype=np.float64),
            device=parent_state.device,
            dtype=parent_state.dtype,
        ).reshape(-1, 1)
        recovery = total_recovery_for_rows(children, state_pos, effective_b)
        q_unit = torch.where(
            effective_b.abs() > 1e-12,
            q_issue / effective_b.clamp_min(1e-12),
            torch.zeros_like(q_issue),
        )

        group["episode"] = episode
        group["production_cashflow"] = production.detach().cpu().reshape(-1).numpy()
        group["debt_adjustment"] = debt_adjustment.detach().cpu().reshape(-1).numpy()
        group["equity_financing_cost"] = equity_cost.detach().cpu().reshape(-1).numpy()
        group["investment_adjustment"] = investment_adjustment.detach().cpu().reshape(-1).numpy()
        group["cashflow_reconstructed"] = cashflow_reconstructed.detach().cpu().reshape(-1).numpy()
        group["cashflow_identity_error"] = (
            exported_cashflow - cashflow_reconstructed
        ).detach().cpu().abs().reshape(-1).numpy()
        group["q_current"] = q_current.detach().cpu().reshape(-1).numpy()
        group["q_unit"] = q_unit.detach().cpu().reshape(-1).numpy()
        group["recovery_grid_mean"] = recovery.detach().cpu().reshape(-1).numpy()
        group["q_minus_recovery"] = (
            q_issue - recovery
        ).detach().cpu().reshape(-1).numpy()
        rows.append(group)

    cashflow_long = pd.concat(rows, ignore_index=True)
    cashflow_long = cashflow_long[
        [
            "episode",
            "source_index",
            "branch",
            "bp_candidate",
            "cashflow",
            "production_cashflow",
            "debt_adjustment",
            "equity_financing_cost",
            "investment_adjustment",
            "cashflow_reconstructed",
            "cashflow_identity_error",
            "q_current",
            "q_issue",
            "q_unit",
            "p_child_mean",
            "default_mean",
            "recovery_grid_mean",
            "q_minus_recovery",
        ]
    ]
    cashflow_summary = (
        cashflow_long.groupby(["episode", "branch"])
        .agg(
            n_rows=("source_index", "size"),
            n_states=("source_index", "nunique"),
            cashflow_mean=("cashflow", "mean"),
            production_cashflow_mean=("production_cashflow", "mean"),
            debt_adjustment_mean=("debt_adjustment", "mean"),
            equity_financing_cost_mean=("equity_financing_cost", "mean"),
            investment_adjustment_mean=("investment_adjustment", "mean"),
            q_current_mean=("q_current", "mean"),
            q_issue_mean=("q_issue", "mean"),
            recovery_grid_mean=("recovery_grid_mean", "mean"),
            q_minus_recovery_mean=("q_minus_recovery", "mean"),
            max_cashflow_identity_error=("cashflow_identity_error", "max"),
        )
        .reset_index()
    )
    cashflow_long.to_csv(output_dir / f"ep{episode}_cashflow_components_long.csv", index=False)
    cashflow_summary.to_csv(output_dir / f"ep{episode}_cashflow_components_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    Config.DEVICE = device

    run_root = args.run_root.resolve()
    firm_pkl = require_file(args.firm_pkl, "firm pickle")
    decomp_long = require_file(args.decomposition_long, "decomposition long CSV")
    decomp_summary = require_file(args.decomposition_summary, "decomposition summary CSV")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_root / "checkpoints"
    policy_ckpt = checkpoint_dir / f"ep{args.episode}_policy_value.pt"
    require_file(policy_ckpt, "policy/value checkpoint")

    models = build_models(
        device=device,
        ckpt_dir=checkpoint_dir,
        ckpt_prefix=f"ep{args.episode}",
        strict=True,
    )
    online_model = models["policy_value"]
    online_model.eval()

    target_model = copy.deepcopy(online_model).to(device)
    target_model.eval()
    for parameter in target_model.parameters():
        parameter.requires_grad_(False)

    hp = build_hyperparams()
    hp.pv_eta_resample_enabled = False
    hp.max_firm_train_units = 0

    summary_df = pd.read_csv(decomp_summary)
    long_df = pd.read_csv(decomp_long)

    tensors = make_episode_batches(
        firm_pkl=firm_pkl,
        model=online_model,
        hp=hp,
        device=device,
        batch_size=args.batch_size,
        n_branches=args.n_branches,
    )
    selected = select_exported_states(summary_df, tensors)

    print(f"Using policy checkpoint: {policy_ckpt}")
    print(f"Using firm data: {firm_pkl}")
    print(f"Using decomposition long: {decomp_long}")
    print(f"Using decomposition summary: {decomp_summary}")
    print(f"Selected states: {selected['parent'].shape[0]}")

    export_logits_and_gradients(
        model=online_model,
        parent_state=selected["parent"],
        source_index=selected["source_index"],
        summary_df=summary_df,
        episode=args.episode,
        output_dir=output_dir,
    )
    export_cashflow_components(
        model=target_model,
        parent_state=selected["parent"],
        children=selected["children"],
        long_df=long_df,
        episode=args.episode,
        output_dir=output_dir,
    )

    print("Saved:")
    print(output_dir / f"ep{args.episode}_bp_logits_states.csv")
    print(output_dir / f"ep{args.episode}_bp_gradient_summary.csv")
    print(output_dir / f"ep{args.episode}_cashflow_components_long.csv")
    print(output_dir / f"ep{args.episode}_cashflow_components_summary.csv")


if __name__ == "__main__":
    main()
