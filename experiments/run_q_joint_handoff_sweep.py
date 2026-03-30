"""
Dedicated Q-stage -> PV/BP-stage handoff sweep runner.

This script is specialized for the "handoff stability" test:
- episode 0 only
- mode0 only
- policy_value only
- fresh initialization for each q-only epoch setting

For each sweep point it exports:
- q_stage_end artifacts
- pvbp_stage_end artifacts
- per-state Q/q_unit/P/bar_z curves as JSON
- aggregate metrics as CSV/JSON
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from models import PolicyValueModel  # noqa: E402
from training.episode import Episode  # noqa: E402
from experiments.run_utils import (  # noqa: E402
    build_hyperparams,
    build_optimizers,
    build_policy_ref_state,
    ensure_dirs,
    plot_bp_diagnostic_curves,
    plot_distributions,
    plot_surfaces,
    save_stage_df,
)


STATE_MAP: List[Tuple[str, float, float]] = [
    ("safe", 0.10, 1.50),
    ("mid", 0.35, 0.50),
    ("risky", 0.60, -0.50),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Q-stage epochs for PV/BP handoff stability")
    parser.add_argument("--run-root", type=Path, default=None, help="Override output root")
    parser.add_argument(
        "--q-stage-epochs-list",
        "--q-only-epochs-list",
        dest="q_stage_epochs_list",
        type=str,
        default="20,40,60,80,100",
        help="Comma-separated Q-stage epochs to sweep",
    )
    parser.add_argument(
        "--pvbp-stage-epochs",
        "--joint-epochs",
        dest="pvbp_stage_epochs",
        type=int,
        default=100,
        help="PV/BP-stage epochs after Q-stage",
    )
    parser.add_argument("--n-samples", type=int, default=None, help="Override sample count")
    parser.add_argument("--n-paths", type=int, default=500, help="Override sample paths")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cuda:0 or cpu")
    parser.add_argument("--log-interval", type=int, default=50, help="Training log interval")
    parser.add_argument("--grid-points", type=int, default=101, help="bp grid size for exported curves")
    parser.add_argument(
        "--states",
        type=str,
        default="safe,mid,risky",
        help="Comma-separated diagnostic states drawn from: safe,mid,risky",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip figure export and only write numeric outputs",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use a smaller sweep-friendly configuration",
    )
    return parser.parse_args()


def parse_int_list(text: str) -> List[int]:
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Q-stage epoch list is empty")
    return values


def make_run_root(arg_path: Path | None) -> Path:
    if arg_path is not None:
        return arg_path.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (ROOT / "cachedir" / f"dl_tensor_q_handoff_sweep_{timestamp}").resolve()


def materialize_episode_outputs(episode: Episode) -> None:
    if episode.df is None and episode.tensor_firm is not None:
        episode.df = episode._table_to_dataframe(episode.tensor_firm)
    if episode.df_macro is None and episode.tensor_macro is not None:
        episode.df_macro = episode._table_to_dataframe(episode.tensor_macro)
    if episode.df_sdf is None and episode.tensor_sdf is not None:
        episode.df_sdf = episode._table_to_dataframe(episode.tensor_sdf)


def make_json_safe(value):
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolve_states(enabled: str) -> List[Tuple[str, float, float]]:
    chosen = {s.strip() for s in str(enabled).split(",") if s.strip()}
    states = [item for item in STATE_MAP if item[0] in chosen]
    return states or [STATE_MAP[0]]


def get_q_unit(model: PolicyValueModel, firm_state: torch.Tensor, out_q: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "get_q_unit"):
        return model.get_q_unit(firm_state)
    return out_q / torch.clamp_min(firm_state[:, 0:1], 1e-6)


def compute_q_curve_exports(
    pv_model: PolicyValueModel,
    ref_state: Dict[str, float],
    device: torch.device,
    grid_points: int,
    enabled_states: List[Tuple[str, float, float]],
    stage_tag: str,
    q_stage_epochs: int,
    pvbp_stage_epochs: int,
) -> Tuple[List[Dict], Dict]:
    metrics: List[Dict] = []
    curves_payload: Dict[str, Dict] = {}
    bp_grid = torch.linspace(0.0, 1.0, grid_points, device=device).unsqueeze(-1)

    for label, b_val, z_val in enabled_states:
        parent = torch.tensor(
            [[
                b_val,
                z_val,
                ref_state["eta"],
                ref_state["i"],
                ref_state["x"],
                ref_state["hatcf"],
                ref_state["lnkf"],
            ]],
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            parent_out = pv_model(parent)
            child = parent.repeat(bp_grid.shape[0], 1)
            child[:, 0:1] = bp_grid
            child_out = pv_model(child)
            q = child_out.Q.squeeze(-1)
            q_unit = get_q_unit(pv_model, child, child_out.Q).squeeze(-1)
            p = child_out.P.squeeze(-1)
            bar_z = child_out.bar_z.squeeze(-1)

        bp_np = bp_grid.squeeze(-1).detach().cpu().numpy()
        q_np = q.detach().cpu().numpy()
        q_unit_np = q_unit.detach().cpu().numpy()
        p_np = p.detach().cpu().numpy()
        bar_z_np = bar_z.detach().cpu().numpy()

        dq_unit = np.diff(q_unit_np)
        q_argmax_idx = int(np.argmax(q_np))
        survive_mask = (p_np > 1e-8) & (bar_z_np < 0.5)
        default_idx = np.where(~survive_mask)[0]
        first_default_bp = float(bp_np[default_idx[0]]) if len(default_idx) > 0 else None

        metrics.append(
            {
                "stage": stage_tag,
                "state": label,
                "q_stage_epochs": q_stage_epochs,
                "pvbp_stage_epochs": pvbp_stage_epochs,
                "q_only_epochs": q_stage_epochs,
                "parent_P": float(parent_out.P.item()),
                "parent_bar_z": float(parent_out.bar_z.item()),
                "q_max": float(np.max(q_np)),
                "q_argmax_bp": float(bp_np[q_argmax_idx]),
                "q_end": float(q_np[-1]),
                "q_drop_from_peak": float(np.max(q_np) - q_np[-1]),
                "q_unit_start": float(q_unit_np[min(1, len(q_unit_np) - 1)]),
                "q_unit_end": float(q_unit_np[-1]),
                "q_unit_monotone_violations": int(np.sum(dq_unit > 1e-4)),
                "q_unit_max_upstep": float(max(0.0, np.max(dq_unit))) if dq_unit.size > 0 else 0.0,
                "survival_share": float(np.mean(survive_mask)),
                "first_default_bp": first_default_bp,
            }
        )

        curves_payload[label] = {
            "parent": {
                "b": b_val,
                "z": z_val,
                "P": float(parent_out.P.item()),
                "bar_z": float(parent_out.bar_z.item()),
            },
            "bp": bp_np.tolist(),
            "Q": q_np.tolist(),
            "q_unit": q_unit_np.tolist(),
            "P": p_np.tolist(),
            "bar_z": bar_z_np.tolist(),
        }

    return metrics, curves_payload


def save_policy_value_checkpoint(models: Dict[str, torch.nn.Module], run_dir: Path, stage_tag: str) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(models["policy_value"].state_dict(), ckpt_dir / f"ep0_{stage_tag}_policy_value.pt")
    torch.save(models["policy_value"].q_model.state_dict(), ckpt_dir / f"ep0_{stage_tag}_policy_value_q.pt")
    torch.save(models["policy_value"].pvbp_model.state_dict(), ckpt_dir / f"ep0_{stage_tag}_policy_value_pvbp.pt")


def export_stage_artifacts(
    episode: Episode,
    models: Dict[str, torch.nn.Module],
    hyperparams,
    device: torch.device,
    run_dir: Path,
    stage_tag: str,
    stage_summary: Dict,
    grid_points: int,
    enabled_states: List[Tuple[str, float, float]],
    q_stage_epochs: int,
    pvbp_stage_epochs: int,
    skip_plots: bool,
) -> List[Dict]:
    materialize_episode_outputs(episode)
    save_policy_value_checkpoint(models, run_dir, stage_tag)

    metrics_dir = run_dir / "data" / "outputs"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if episode.df is not None and not episode.df.empty:
        save_stage_df(0, stage_tag, run_dir, episode.df, episode.df_macro, episode.df_sdf)
        ref_state = build_policy_ref_state(episode.df)
        if not skip_plots:
            plot_surfaces(
                0,
                models["policy_value"],
                None,
                ref_state,
                device,
                run_dir,
                tag=stage_tag,
            )
            plot_bp_diagnostic_curves(
                0,
                models["policy_value"],
                None,
                ref_state,
                device,
                run_dir,
                hyperparams=hyperparams,
                tag=stage_tag,
            )
            plot_distributions(
                0,
                episode.df,
                models["policy_value"],
                device,
                run_dir,
                df_macro=episode.df_macro,
                tag=stage_tag,
            )
    else:
        ref_state = {
            "eta": 1.0,
            "i": 0.25,
            "x": Config.XBAR,
            "hatcf": -2.0,
            "lnkf": 4.0,
        }

    metrics, curves = compute_q_curve_exports(
        pv_model=models["policy_value"],
        ref_state=ref_state,
        device=device,
        grid_points=grid_points,
        enabled_states=enabled_states,
        stage_tag=stage_tag,
        q_stage_epochs=q_stage_epochs,
        pvbp_stage_epochs=pvbp_stage_epochs,
    )

    with (metrics_dir / f"ep0_{stage_tag}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "stage_tag": stage_tag,
                "stage_summary": make_json_safe(stage_summary),
                "curve_metrics": make_json_safe(metrics),
                "curves": make_json_safe(curves),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return metrics


def configure_hyperparams(args: argparse.Namespace, q_stage_epochs: int):
    hp = build_hyperparams()
    if args.quick_test:
        hp.n_samples = min(hp.n_samples, 1000)
        hp.n_paths = min(args.n_paths, 100)
        hp.batch_size = min(hp.batch_size, 1024)
    else:
        hp.n_paths = args.n_paths
    if args.n_samples is not None:
        hp.n_samples = args.n_samples
    if args.batch_size is not None:
        hp.batch_size = args.batch_size
    hp.q_stage_epochs = max(100, q_stage_epochs)
    hp.pvbp_stage_epochs = max(100, int(args.pvbp_stage_epochs))
    hp.q_pretrain_epochs = hp.q_stage_epochs
    hp.q_warmstart_epochs = hp.q_stage_epochs
    hp.epochs = hp.q_stage_epochs + hp.pvbp_stage_epochs
    hp.bp_diag_enabled = True
    hp.bp_diag_every_n_episodes = 1
    hp.bp_diag_states = ",".join([s[0] for s in resolve_states(args.states)])
    return hp


def main():
    args = parse_args()
    run_root = make_run_root(args.run_root)
    ensure_dirs(run_root)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    Config.DEVICE = device

    q_stage_list = parse_int_list(args.q_stage_epochs_list)
    enabled_states = resolve_states(args.states)

    print(f"Run root: {run_root}")
    print(f"Device: {device}")
    print(f"Q-stage sweep: {q_stage_list}")
    print(f"PV/BP-stage epochs: {args.pvbp_stage_epochs}")
    print(f"States: {[s[0] for s in enabled_states]}")
    print(f"Skip plots: {int(args.skip_plots)}")

    all_metrics: List[Dict] = []
    run_manifest: List[Dict] = []

    for q_stage_epochs in q_stage_list:
        pvbp_stage_epochs = int(args.pvbp_stage_epochs)
        run_dir = run_root / f"qstage_{q_stage_epochs:03d}_pvbp_{pvbp_stage_epochs:03d}"
        ensure_dirs(run_dir)
        print(f"\n=== Sweep run: q_stage={q_stage_epochs}, pvbp_stage={pvbp_stage_epochs} ===")
        print(f"Run dir: {run_dir}")

        hyperparams = configure_hyperparams(args, q_stage_epochs)
        models = {"policy_value": PolicyValueModel().to(device)}
        optimizers = build_optimizers(models, hyperparams)
        episode = Episode(
            models=models,
            optimizers=optimizers,
            config=Config,
            hyperparams=hyperparams,
            device=device,
            episode_id=0,
        )

        stage_exports: List[Dict] = []

        def on_policy_stage(stage_summary: Dict):
            phase_to_tag = {
                "q_stage_end": "q_stage_end",
                "pvbp_stage_end": "pvbp_stage_end",
                "policy_end": "policy_end",
            }
            stage_tag = phase_to_tag.get(stage_summary.get("phase", "policy_stage"), str(stage_summary.get("phase", "policy_stage")))
            stage_metrics = export_stage_artifacts(
                episode=episode,
                models=models,
                hyperparams=hyperparams,
                device=device,
                run_dir=run_dir,
                stage_tag=stage_tag,
                stage_summary=stage_summary,
                grid_points=args.grid_points,
                enabled_states=enabled_states,
                q_stage_epochs=q_stage_epochs,
                pvbp_stage_epochs=pvbp_stage_epochs,
                skip_plots=args.skip_plots,
            )
            all_metrics.extend(stage_metrics)
            stage_exports.append(
                {
                    "stage": stage_tag,
                    "epoch": stage_summary.get("epoch"),
                    "metrics_rows": len(stage_metrics),
                }
            )

        summary = episode.run_episode(
            n_epochs=hyperparams.epochs,
            batch_size=hyperparams.batch_size,
            log_interval=args.log_interval,
            n_samples=hyperparams.n_samples,
            n_paths=hyperparams.n_paths,
            group_size=2,
            n_branches=Config.BRANCH_NUM,
            train_modules=["policy_value"],
            simulate_kwargs={"horizon_mode1": 1, "horizon": hyperparams.simulate_horizon},
            episode_mode="mode0",
            policy_stage_callback=on_policy_stage,
        )
        materialize_episode_outputs(episode)
        save_stage_df(0, "mode0", run_dir, episode.df, episode.df_macro, episode.df_sdf)

        module_summaries = summary.get("module_summaries", summary)
        run_payload = {
            "q_stage_epochs": q_stage_epochs,
            "pvbp_stage_epochs": pvbp_stage_epochs,
            "q_only_epochs": q_stage_epochs,
            "run_dir": str(run_dir),
            "stage_exports": stage_exports,
            "module_summaries": make_json_safe(module_summaries),
        }
        with (run_dir / "sweep_summary.json").open("w", encoding="utf-8") as f:
            json.dump(run_payload, f, indent=2, ensure_ascii=False)
        run_manifest.append(run_payload)

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.sort_values(["q_stage_epochs", "stage", "state"], inplace=True)
        metrics_df.to_csv(run_root / "handoff_metrics.csv", index=False)
        with (run_root / "handoff_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(make_json_safe(metrics_df.to_dict(orient="records")), f, indent=2, ensure_ascii=False)

        summary_cols = [
            "q_stage_epochs",
            "stage",
            "state",
            "parent_P",
            "parent_bar_z",
            "q_max",
            "q_argmax_bp",
            "q_drop_from_peak",
            "q_unit_monotone_violations",
            "survival_share",
            "first_default_bp",
        ]
        print("\n=== Handoff metric summary ===")
        print(metrics_df[summary_cols].to_string(index=False))

    with (run_root / "sweep_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(run_manifest), f, indent=2, ensure_ascii=False)

    print(f"\nSweep finished. Outputs written to: {run_root}")


if __name__ == "__main__":
    main()
