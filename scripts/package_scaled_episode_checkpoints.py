"""Package scaled raw episode state_dict checkpoints for strict analysis loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.checkpoint_loader import load_analysis_checkpoint
from analysis.economic_config import AnalysisEconomicConfig
from experiments.run_utils import build_hyperparams, build_models
from models import PolicyValueModel


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_episodes(text: Optional[str]) -> Optional[list[int]]:
    if not text:
        return None
    return [int(part) for part in text.replace(",", " ").split()]


def _episode_from_policy_path(path: Path) -> int:
    stem = path.name.split("_", 1)[0]
    if not stem.startswith("ep"):
        raise ValueError(f"Cannot parse episode from checkpoint name: {path.name}")
    return int(stem[2:])


def _policy_paths(source_dir: Path, episodes: Optional[Iterable[int]]) -> list[Path]:
    if episodes is None:
        paths = sorted(source_dir.glob("ep*_policy_value.pt"), key=_episode_from_policy_path)
    else:
        paths = [source_dir / f"ep{int(ep)}_policy_value.pt" for ep in episodes]
    if not paths:
        raise RuntimeError(f"No ep*_policy_value.pt checkpoints found in {source_dir}")
    return paths


def _configured_hyperparams(args: argparse.Namespace) -> dict:
    hp = build_hyperparams()
    hp.pv_value_scale_mode = "exp_xz"
    hp.pv_value_scale_log_max = float(args.value_scale_log_max)
    hp.pv_bellman_normalize_by_value_scale = True

    hp.pv_training_flow = str(args.pv_training_flow)
    hp.pv_use_clipped_m = bool(args.pv_use_clipped_m)
    hp.pv_m_clamp_min = float(args.pv_m_clamp_min)
    hp.pv_m_clamp_max = float(args.pv_m_clamp_max)
    hp.q_use_detached_m = bool(args.q_use_detached_m)

    hp.bp_grid_policy_loss_space = str(args.bp_grid_policy_loss_space)
    hp.bp_grid_logit_target_eps = float(args.bp_grid_logit_target_eps)
    hp.bp_grid_logit_huber_delta = float(args.bp_grid_logit_huber_delta)
    hp.pv_mixture_enabled = bool(args.pv_mixture_enabled)
    hp.pv_mixture_ratio = float(args.pv_mixture_ratio)
    hp.pv_mixture_start_episode = int(args.pv_mixture_start_episode)
    hp.pv_mixture_budget_mode = str(args.pv_mixture_budget_mode)
    hp.pv_mixture_sampling_mode = str(args.pv_mixture_sampling_mode)
    hp.pv_mixture_coverage_group_size = int(args.pv_mixture_coverage_group_size)
    hp.pv_mixture_seed = int(args.pv_mixture_seed)
    hp.pv_mixture_stratified_validation = bool(args.pv_mixture_stratified_validation)
    hp.pv_mixture_preserve_rng = bool(args.pv_mixture_preserve_rng)
    hp.firm_target_update = str(args.firm_target_update)
    return dict(hp.__dict__)


def package_episode_checkpoints(args: argparse.Namespace) -> list[Path]:
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    hyperparams_payload = _configured_hyperparams(args)
    config_snapshot = AnalysisEconomicConfig.from_current_config().to_dict()
    value_scale_log_max = float(args.value_scale_log_max)
    value_parameterization = {
        "mode": "exp_xz",
        "scale_formula": f"1+exp(clamp(x+z,max={value_scale_log_max:g}))",
        "bellman_normalization": True,
        "log_max": value_scale_log_max,
    }

    outputs: list[Path] = []
    for policy_path in _policy_paths(source_dir, _parse_episodes(args.episodes)):
        if not policy_path.exists():
            raise RuntimeError(f"Missing policy checkpoint: {policy_path}")
        ep = _episode_from_policy_path(policy_path)
        sdf_path = source_dir / f"ep{ep}_sdf_fc1.pt"
        if not sdf_path.exists():
            raise RuntimeError(f"Missing SDF/FC1 checkpoint for episode {ep}: {sdf_path}")

        policy_state = torch.load(policy_path, map_location=device)
        sdf_state = torch.load(sdf_path, map_location=device)

        policy_model = PolicyValueModel(
            value_scale_mode="exp_xz",
            value_scale_log_max=value_scale_log_max,
        ).to(device)
        policy_model.load_state_dict(policy_state, strict=True)

        models = build_models(device)
        models["sdf_fc1"].load_state_dict(sdf_state, strict=True)

        payload = {
            "models": {
                "policy_value": policy_state,
                "sdf_fc1": sdf_state,
            },
            "hyperparams": hyperparams_payload,
            "config_snapshot": config_snapshot,
            "policy_value_model_spec": policy_model.model_spec(),
            "value_parameterization": value_parameterization,
            "analysis_packaging": {
                "episode": ep,
                "source_policy_checkpoint": str(policy_path),
                "source_sdf_checkpoint": str(sdf_path),
                "source_policy_sha256": _sha256(policy_path),
                "source_sdf_sha256": _sha256(sdf_path),
                "training_commit": str(args.training_commit),
                "source_run_root": str(args.run_root) if args.run_root else None,
            },
        }

        output_path = output_dir / f"ep{ep}_combined.pt"
        torch.save(payload, output_path)
        outputs.append(output_path)
        print(f"saved: {output_path}")

    packaged_eps = sorted({_episode_from_policy_path(path) for path in outputs})
    smoke_eps = sorted({packaged_eps[0], packaged_eps[-1]})
    for ep in smoke_eps:
        path = output_dir / f"ep{ep}_combined.pt"
        loaded = load_analysis_checkpoint(path, device="cpu")
        meta = loaded.metadata["value_parameterization"]["checkpoint"]
        if meta != value_parameterization:
            raise RuntimeError(f"strict load metadata mismatch for {path}: {meta} != {value_parameterization}")
        print(f"strict_load_smoke ep{ep}: {json.dumps(meta, sort_keys=True)}")

    print(f"combined checkpoint directory: {output_dir}")
    return outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Package scaled raw episode checkpoints into combined analysis checkpoints.")
    p.add_argument("--run-root")
    p.add_argument("--source-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--episodes", default="")
    p.add_argument("--training-commit", required=True)
    p.add_argument("--value-scale-log-max", type=float, default=20.0)
    p.add_argument("--pv-training-flow", default="staged")
    p.add_argument("--pv-use-clipped-m", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pv-m-clamp-min", type=float, default=0.7)
    p.add_argument("--pv-m-clamp-max", type=float, default=1.3)
    p.add_argument("--q-use-detached-m", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bp-grid-policy-loss-space", default="logit")
    p.add_argument("--bp-grid-logit-target-eps", type=float, default=1e-4)
    p.add_argument("--bp-grid-logit-huber-delta", type=float, default=1.0)
    p.add_argument("--pv-mixture-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pv-mixture-ratio", type=float, default=0.20)
    p.add_argument("--pv-mixture-start-episode", type=int, default=1)
    p.add_argument("--pv-mixture-budget-mode", default="fixed_total")
    p.add_argument("--pv-mixture-sampling-mode", default="uniform")
    p.add_argument("--pv-mixture-coverage-group-size", type=int, default=2)
    p.add_argument("--pv-mixture-seed", type=int, default=24680)
    p.add_argument("--pv-mixture-stratified-validation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pv-mixture-preserve-rng", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--firm-target-update", default="stage_hard")
    return p.parse_args()


def main() -> None:
    package_episode_checkpoints(parse_args())


if __name__ == "__main__":
    main()
