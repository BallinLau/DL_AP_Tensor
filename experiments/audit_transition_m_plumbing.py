#!/usr/bin/env python3
"""Read-only transition-M plumbing audit on the historical inner bank."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.economic_config import AnalysisEconomicConfig  # noqa: E402
from config import Config, HyperParams  # noqa: E402
from experiments.run_utils import build_optimizers, load_training_warmstart  # noqa: E402
from losses.q_loss import classify_q_parent_regimes  # noqa: E402
from training.episode import Episode  # noqa: E402
from training.inner_fixed_point import (  # noqa: E402
    InnerFPCollocationBank,
    InnerFPChildBank,
    evaluate_inner_operator_snapshot,
    make_batches,
    write_json,
)


DEFAULT_ARTIFACT = Path(
    "/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/artifacts/"
    "inner_fp_block_isolation_20261015/p_only"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "cachedir" / "transition_m_plumbing_fix_20261015",
    )
    parser.add_argument("--parents", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(value: torch.Tensor) -> Dict[str, float]:
    flat = value.detach().reshape(-1).cpu()
    return {
        "mean": float(flat.mean().item()),
        "p50": float(torch.quantile(flat, 0.50).item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
        "p99": float(torch.quantile(flat, 0.99).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "nonunit_share": float((flat.sub(1.0).abs() > 1e-6).float().mean().item()),
        "max_abs_from_one": float(flat.sub(1.0).abs().max().item()),
    }


def load_episode(checkpoint: Path, device: torch.device) -> Episode:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    economic = AnalysisEconomicConfig.from_dict(payload["config_snapshot"])
    for name, value in economic.to_dict().items():
        setattr(Config, name, value)
    Config.DEVICE = device
    hp = HyperParams()
    for name, value in payload["hyperparams"].items():
        setattr(hp, name, value)
    hp.bp_distill_epochs = 0
    hp.pv_bp_head_training_enabled = False
    value = payload["value_parameterization"]
    models, firm_target, _ = load_training_warmstart(
        checkpoint,
        device,
        expected_episode=1,
        expected_q_parameterization="direct",
        expected_value_scale_mode=str(value["mode"]),
        expected_value_scale_log_max=float(value["log_max"]),
        expected_bellman_normalization=bool(value["bellman_normalization"]),
    )
    return Episode(
        models=models,
        optimizers=build_optimizers(models, hp),
        config=Config,
        hyperparams=hp,
        device=device,
        episode_id=1,
        firm_target=firm_target,
        q_checkpoint_loaded=True,
    )


def load_banks(root: Path, n: int) -> tuple[InnerFPCollocationBank, InnerFPChildBank]:
    parent_payload = torch.load(
        root / "fixed_point" / "collocation_val.pt",
        map_location="cpu",
        weights_only=False,
    )
    child_payload = torch.load(
        root / "fixed_point" / "child_shock_bank.pt",
        map_location="cpu",
        weights_only=False,
    )["validation"]
    count = min(int(n), int(parent_payload["parent_states"].shape[0]))
    parent = InnerFPCollocationBank(
        parent_states=parent_payload["parent_states"][:count].clone(),
        source_kind=parent_payload["source_kind"][:count].clone(),
        seed=int(parent_payload["seed"]),
        split="transition_m_audit",
        metadata={**parent_payload.get("metadata", {}), "audit_subset": count},
    )
    child = InnerFPChildBank(
        children=child_payload["children"][:count].clone(),
        eps_x=child_payload["eps_x"][:count].clone(),
        eps_z=child_payload["eps_z"][:count].clone(),
        u_eta=child_payload["u_eta"][:count].clone(),
        u_i=child_payload["u_i"][:count].clone(),
        seed=int(child_payload["seed"]),
        metadata={**child_payload.get("metadata", {}), "audit_subset": count},
    )
    return parent, child


def q_target_components(
    episode: Episode,
    batch: Dict[str, torch.Tensor],
    used_m: List[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    model = episode.models["policy_value"]
    state = batch["parent"][:, :7]
    children = episode._get_policy_children(batch)
    output = model(state)
    bar_i = episode._policy_get_out(output, "bar_i", 5)
    multiplier = bar_i * (float(episode.loss_fns["q"].g) - 1.0) + 1.0
    b_sp = state[:, 0:1] / multiplier.clamp_min(1e-6)
    q_sp: List[torch.Tensor] = []
    bar_z: List[torch.Tensor] = []
    for child in children:
        child_state = child[:, :7].clone()
        child_state[:, 0:1] = b_sp
        child_output = model(child_state)
        q_sp.append(episode._policy_get_out(child_output, "Q", 0))
        bar_z.append(episode._policy_get_out(child_output, "bar_z", 6))
    components = episode.loss_fns["q"].compute_main_components(
        model._q_claim_output(state),
        state[:, 0:1],
        bar_i,
        used_m,
        q_sp,
        bar_z,
        [child[:, 4:5] for child in children],
        [child[:, 1:2] for child in children],
    )
    keys = ("q_target_total", "continuation_q_component", "recursion_gain")
    return {
        key: torch.stack([item[key] for item in components], dim=1)
        for key in keys
    }


def main() -> int:
    args = parse_args()
    root = args.artifact_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "source" / "checkpoints" / "ep1_combined.pt"
    device = torch.device(args.device)
    episode = load_episode(checkpoint, device)
    parent_bank, child_bank = load_banks(root, args.parents)
    batches = make_batches(parent_bank, child_bank, batch_size=args.parents, device=device)
    batch = batches[0]
    children = episode._get_policy_children(batch)

    p_lo = float(getattr(episode.hyperparams, "pv_m_clamp_min", 0.7))
    p_hi = float(getattr(episode.hyperparams, "pv_m_clamp_max", 1.3))
    p_raw, p_used, p_source = episode._build_policy_m_lists(
        batch["parent"], children, p_lo, p_hi,
        allow_unit_fallback=False, return_source=True,
    )
    q_raw, q_used, q_source = episode._build_q_m_lists(
        children, int(batch["parent"].shape[0]),
        allow_unit_fallback=False, return_source=True,
    )

    inner_cache, inner_cache_summary = episode._build_pq_value_target_cache(
        [batch], episode.models["policy_value"],
        q_target_model=episode.models["policy_value"],
    )
    outer_batch = dict(batch)
    outer_batch["parent"] = torch.cat(
        [batch["parent"], torch.full_like(batch["parent"][:, :1], 999.0)], dim=1
    )
    outer_cache, outer_cache_summary = episode._build_pq_value_target_cache(
        [outer_batch], episode.models["policy_value"],
        q_target_model=episode.models["policy_value"],
    )

    with torch.no_grad():
        q_train = q_target_components(episode, batch, q_used)
        _raw_eval, q_eval_used, _source_eval = episode._build_q_m_lists(
            children, int(batch["parent"].shape[0]),
            allow_unit_fallback=False, return_source=True,
        )
        q_eval = q_target_components(episode, batch, q_eval_used)

    corrected_metrics, _, corrected_snapshot = evaluate_inner_operator_snapshot(
        episode, batches
    )
    legacy_children = deepcopy(child_bank)
    legacy_children.children[:, :, 7] = 1.0
    legacy_batches = make_batches(
        parent_bank, legacy_children, batch_size=args.parents, device=device
    )
    legacy_metrics, _, legacy_snapshot = evaluate_inner_operator_snapshot(
        episode, legacy_batches
    )

    def residual(snapshot: Dict[str, torch.Tensor], name: str) -> Dict[str, float]:
        error = (snapshot[f"{name}_target"] - snapshot[f"{name}_value"]).abs()
        normalized = error / snapshot[f"{name}_scale"].clamp_min(1e-12)
        if name == "q":
            normalized = normalized[snapshot["q_regime"].reshape(-1) == 2]
        flat = normalized.reshape(-1)
        return {
            "mean": float(flat.mean().item()),
            "p90": float(torch.quantile(flat, 0.90).item()),
            "n": int(flat.numel()),
        }

    historical_files = [
        root / "fixed_point" / "metrics.json",
        root / "fixed_point" / "final_summary.json",
        root / "fixed_point" / "block_isolation_metrics.csv",
        root / "checkpoints" / "final.pt",
    ]
    result = {
        "definition": "transition_m_plumbing_read_only_audit_v1",
        "parents": int(parent_bank.parent_states.shape[0]),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_sha256(checkpoint),
        "m_source": {"p": p_source, "q": q_source},
        "raw_m": stats(torch.cat([value.reshape(-1) for value in p_raw])),
        "p_used_m": stats(torch.cat([value.reshape(-1) for value in p_used])),
        "q_used_m": stats(torch.cat([value.reshape(-1) for value in q_used])),
        "p_outer_inner_parity": {
            "p0_target_max_abs_diff": float(
                (outer_cache[0].p0_value_target - inner_cache[0].p0_value_target)
                .abs().max().item()
            ),
            "pi_target_max_abs_diff": float(
                (outer_cache[0].pi_value_target - inner_cache[0].pi_value_target)
                .abs().max().item()
            ),
            "outer_m_hash": outer_cache[0].m_hash,
            "inner_m_hash": inner_cache[0].m_hash,
            "outer_m_source": outer_cache_summary["p_m_source"],
            "inner_m_source": inner_cache_summary["p_m_source"],
        },
        "q_train_eval_parity": {
            key + "_max_abs_diff": float((q_train[key] - q_eval[key]).abs().max().item())
            for key in q_train
        },
        "legacy_vs_corrected_normalized_residual": {
            name: {
                "legacy_m1": residual(legacy_snapshot, name),
                "corrected_child_m": residual(corrected_snapshot, name),
            }
            for name in ("p0", "pi", "q")
        },
        "corrected_transition_m_metrics": corrected_metrics["transition_m"],
        "smoke": {
            "p_target_builder_m_source": inner_cache_summary["p_m_source"],
            "q_survival_builder_m_source": q_source,
            "p_used_m_identically_one": bool(
                torch.equal(torch.cat(p_used), torch.ones_like(torch.cat(p_used)))
            ),
            "q_used_m_identically_one": bool(
                torch.equal(torch.cat(q_used), torch.ones_like(torch.cat(q_used)))
            ),
            "optimizer_steps": 0,
        },
        "historical_artifact_sha256": {
            str(path.relative_to(root)): file_sha256(path) for path in historical_files
        },
    }
    write_json(output / "audit.json", result)
    write_json(output / "smoke.json", result["smoke"])
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"TRANSITION_M_AUDIT_DIR={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
