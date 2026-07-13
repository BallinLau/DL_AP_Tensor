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
from experiments.run_utils import build_hyperparams, build_models  # noqa: E402
from losses import P0Loss, PILoss  # noqa: E402
from training.bp_grid_teacher import BPGridTeacher  # noqa: E402
from training.episode import Episode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export target-grid Bellman objective decompositions for selected "
            "active refinancing firm states."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--firm-pkl", type=Path, required=True)
    parser.add_argument("--n-states", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-branches", type=int, default=2)
    return parser.parse_args()


def cat_batches(batches: Iterable[Dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    values = [batch[key] for batch in batches if batch.get(key) is not None]
    if not values:
        raise RuntimeError(f"No values found for batch key: {key}")
    return torch.cat(values, dim=0)


def require_batch_width(name: str, tensor: torch.Tensor, expected: int = 8) -> None:
    print(f"{name}.shape={tuple(tensor.shape)}")
    if tensor.dim() != 2 or tensor.shape[1] != expected:
        raise RuntimeError(
            f"Expected {name} to have shape (N, {expected}) with 7 state columns plus M, "
            f"got {tuple(tensor.shape)}"
        )


def make_long_rows(
    result: Dict[str, torch.Tensor],
    parent_state: torch.Tensor,
    source_index: torch.Tensor,
    branch: str,
    bp_pred: torch.Tensor,
) -> List[Dict[str, object]]:
    grid = result["coarse_bp_grid"]
    value = result["coarse_value_grid"]
    cashflow = result["coarse_cashflow_grid_mean"]
    continuation = result["coarse_continuation_grid_mean"]
    q_issue = result["coarse_q_issue_grid"]
    p_child = result["coarse_p_child_grid_mean"]
    default = result["coarse_default_grid_mean"]

    rows: List[Dict[str, object]] = []
    bp_pred_flat = bp_pred.reshape(-1)

    for state_pos in range(parent_state.shape[0]):
        for grid_pos in range(grid.shape[1]):
            rows.append(
                {
                    "state_pos": state_pos,
                    "source_index": int(source_index[state_pos].item()),
                    "branch": branch,
                    "grid_index": grid_pos,
                    "b": float(parent_state[state_pos, 0].item()),
                    "z": float(parent_state[state_pos, 1].item()),
                    "eta": float(parent_state[state_pos, 2].item()),
                    "i": float(parent_state[state_pos, 3].item()),
                    "x": float(parent_state[state_pos, 4].item()),
                    "hatcf": float(parent_state[state_pos, 5].item()),
                    "lnkf": float(parent_state[state_pos, 6].item()),
                    "bp_pred": float(bp_pred_flat[state_pos].item()),
                    "bp_candidate": float(grid[state_pos, grid_pos].item()),
                    "cashflow": float(cashflow[state_pos, grid_pos].item()),
                    "continuation": float(continuation[state_pos, grid_pos].item()),
                    "value": float(value[state_pos, grid_pos].item()),
                    "q_issue": float(q_issue[state_pos, grid_pos].item()),
                    "p_child_mean": float(p_child[state_pos, grid_pos].item()),
                    "default_mean": float(default[state_pos, grid_pos].item()),
                }
            )
    return rows


def make_summary_rows(
    result: Dict[str, torch.Tensor],
    parent_state: torch.Tensor,
    source_index: torch.Tensor,
    branch: str,
    bp_pred: torch.Tensor,
) -> List[Dict[str, object]]:
    grid = result["coarse_bp_grid"]
    value = result["coarse_value_grid"]
    cashflow = result["coarse_cashflow_grid_mean"]
    continuation = result["coarse_continuation_grid_mean"]
    q_issue = result["coarse_q_issue_grid"]
    p_child = result["coarse_p_child_grid_mean"]
    default = result["coarse_default_grid_mean"]
    bp_star_teacher = result["bp_star"].reshape(-1)
    value_star_teacher = result["value_star"].reshape(-1)
    regret_at_pred = result["regret"].reshape(-1)
    q_issue_teacher_star = result["q_issue_at_star"].reshape(-1)
    p_child_teacher_star = result["p_child_at_star"].reshape(-1)
    default_teacher_star = result["default_at_star"].reshape(-1)

    argmax = value.argmax(dim=1)
    bp_pred_flat = bp_pred.reshape(-1)
    rows: List[Dict[str, object]] = []

    for state_pos in range(parent_state.shape[0]):
        j_star = int(argmax[state_pos].item())

        value_low = value[state_pos, 0]
        cf_low = cashflow[state_pos, 0]
        cont_low = continuation[state_pos, 0]

        value_star_coarse = value[state_pos, j_star]
        cf_star_coarse = cashflow[state_pos, j_star]
        cont_star_coarse = continuation[state_pos, j_star]

        delta_cf = cf_star_coarse - cf_low
        delta_cont = cont_star_coarse - cont_low
        delta_value = value_star_coarse - value_low
        identity_error = (value[state_pos] - cashflow[state_pos] - continuation[state_pos]).abs().max()

        if abs(float(delta_cf.item())) >= abs(float(delta_cont.item())):
            dominant_component = "cashflow"
        else:
            dominant_component = "continuation"

        rows.append(
            {
                "state_pos": state_pos,
                "source_index": int(source_index[state_pos].item()),
                "branch": branch,
                "b": float(parent_state[state_pos, 0].item()),
                "z": float(parent_state[state_pos, 1].item()),
                "eta": float(parent_state[state_pos, 2].item()),
                "i": float(parent_state[state_pos, 3].item()),
                "x": float(parent_state[state_pos, 4].item()),
                "hatcf": float(parent_state[state_pos, 5].item()),
                "lnkf": float(parent_state[state_pos, 6].item()),
                "bp_pred": float(bp_pred_flat[state_pos].item()),
                "bp_star_coarse": float(grid[state_pos, j_star].item()),
                "bp_star_teacher": float(bp_star_teacher[state_pos].item()),
                "argmax_index": j_star,
                "value_low": float(value_low.item()),
                "value_star_coarse": float(value_star_coarse.item()),
                "value_star_teacher": float(value_star_teacher[state_pos].item()),
                "regret_at_pred": float(regret_at_pred[state_pos].item()),
                "cashflow_low": float(cf_low.item()),
                "cashflow_coarse_star": float(cf_star_coarse.item()),
                "continuation_low": float(cont_low.item()),
                "continuation_coarse_star": float(cont_star_coarse.item()),
                "delta_cashflow_low_to_coarse_star": float(delta_cf.item()),
                "delta_continuation_low_to_coarse_star": float(delta_cont.item()),
                "delta_value_low_to_coarse_star": float(delta_value.item()),
                "q_issue_coarse_star": float(q_issue[state_pos, j_star].item()),
                "p_child_coarse_star": float(p_child[state_pos, j_star].item()),
                "default_coarse_star": float(default[state_pos, j_star].item()),
                "q_issue_teacher_star": float(q_issue_teacher_star[state_pos].item()),
                "p_child_teacher_star": float(p_child_teacher_star[state_pos].item()),
                "default_teacher_star": float(default_teacher_star[state_pos].item()),
                "identity_error_max": float(identity_error.item()),
                "dominant_component": dominant_component,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.n_states <= 0:
        raise ValueError("--n-states must be positive")

    device = torch.device(args.device)
    Config.DEVICE = device

    run_root = args.run_root.resolve()
    firm_pkl = args.firm_pkl.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_root / "data" / "outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not firm_pkl.exists():
        raise FileNotFoundError(f"Firm pickle not found: {firm_pkl}")

    checkpoint_dir = run_root / "checkpoints"
    policy_ckpt = checkpoint_dir / f"ep{args.episode}_policy_value.pt"
    if not policy_ckpt.exists():
        raise FileNotFoundError(
            "Policy/value checkpoint is required for decomposition export; "
            f"missing {policy_ckpt}. Refusing to fall back to a random model."
        )
    print(f"Using policy checkpoint: {policy_ckpt}")
    print(f"Using firm data: {firm_pkl}")

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

    episode = Episode.__new__(Episode)
    episode.device = device
    episode.hyperparams = hp
    episode.models = {"policy_value": online_model}

    df = pd.read_pickle(firm_pkl)
    batches = episode._create_firm_batches_from_df(
        df,
        batch_size=args.batch_size,
        n_branches=args.n_branches,
        eta_resample=False,
    )
    if not batches:
        raise RuntimeError("No parent-child batches were constructed.")

    parent_full = cat_batches(batches, "parent")
    child0_full = cat_batches(batches, "child0")
    child1_full = cat_batches(batches, "child1")
    source_index_full = cat_batches(batches, "parent_source_index")

    require_batch_width("parent_full", parent_full)
    require_batch_width("child0_full", child0_full)
    require_batch_width("child1_full", child1_full)

    parent_state_full = parent_full[:, :7]
    children_full = [child0_full[:, :7], child1_full[:, :7]]
    m_full = [child0_full[:, 7:8], child1_full[:, 7:8]]

    with torch.no_grad():
        online_out = online_model(parent_state_full)
        mix_weight_full = online_out.bar_i_cond.clamp(0.0, 1.0)
        bp_mix_full = (1.0 - mix_weight_full) * online_out.bp0 + mix_weight_full * online_out.bpI

    active_mask = parent_state_full[:, 2] > 0.5
    active_index = torch.where(active_mask)[0]
    if active_index.numel() == 0:
        raise RuntimeError("No active refinancing parent states found.")

    active_bp = bp_mix_full[active_index, 0]
    n_select = min(args.n_states, int(active_index.numel()))
    selected_order = torch.topk(active_bp, k=n_select, largest=True).indices
    selected = active_index[selected_order]

    parent_state = parent_state_full[selected]
    children = [child[selected] for child in children_full]
    m_list = [m[selected] for m in m_full]
    source_index = source_index_full[selected]

    with torch.no_grad():
        selected_out = online_model(parent_state)
        mix_weight = selected_out.bar_i_cond.clamp(0.0, 1.0)
        bp_mix = (1.0 - mix_weight) * selected_out.bp0 + mix_weight * selected_out.bpI

    teacher = BPGridTeacher.from_hyperparams(
        target_model=target_model,
        p0_loss_fn=P0Loss(),
        pi_loss_fn=PILoss(),
        hyperparams=hp,
    )

    branch_inputs = {
        "p0": {"bp_pred": selected_out.bp0, "mix_weight": None},
        "pi": {"bp_pred": selected_out.bpI, "mix_weight": None},
        "mix": {"bp_pred": bp_mix, "mix_weight": mix_weight},
    }

    long_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for branch, inputs in branch_inputs.items():
            result = teacher.compute(
                parent_state,
                children,
                m_list,
                branch=branch,
                bp_pred=inputs["bp_pred"],
                mix_weight=inputs["mix_weight"],
            )
            long_rows.extend(
                make_long_rows(
                    result=result,
                    parent_state=parent_state,
                    source_index=source_index,
                    branch=branch,
                    bp_pred=inputs["bp_pred"],
                )
            )
            summary_rows.extend(
                make_summary_rows(
                    result=result,
                    parent_state=parent_state,
                    source_index=source_index,
                    branch=branch,
                    bp_pred=inputs["bp_pred"],
                )
            )

    long_df = pd.DataFrame(long_rows)
    summary_df = pd.DataFrame(summary_rows)

    long_path = output_dir / f"ep{args.episode}_target_grid_decomposition_long.csv"
    summary_path = output_dir / f"ep{args.episode}_target_grid_decomposition_summary.csv"
    long_df.to_csv(long_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved long decomposition: {long_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Selected active states: {n_select}")
    print("Maximum decomposition identity error:", summary_df["identity_error_max"].max())

    print("\nDominant component counts:")
    print(summary_df.groupby(["branch", "dominant_component"]).size())

    print("\nTop states:")
    display_cols = [
        "source_index",
        "branch",
        "b",
        "bp_pred",
        "bp_star_coarse",
        "bp_star_teacher",
        "regret_at_pred",
        "delta_cashflow_low_to_coarse_star",
        "delta_continuation_low_to_coarse_star",
        "delta_value_low_to_coarse_star",
        "q_issue_coarse_star",
        "p_child_coarse_star",
        "default_coarse_star",
        "q_issue_teacher_star",
        "p_child_teacher_star",
        "default_teacher_star",
        "dominant_component",
    ]
    print(
        summary_df[display_cols]
        .sort_values(["branch", "bp_star_coarse"], ascending=[True, False])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
