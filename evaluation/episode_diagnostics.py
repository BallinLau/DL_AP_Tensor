"""Lightweight, read-only per-episode diagnostics for staged Hybrid-Q runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import torch


REFERENCE_COLUMNS = ("x", "i", "Hatcf", "LnKF")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _module_hash(module: torch.nn.Module, names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    found = False
    for name in names:
        child = getattr(module, name, None)
        if child is None:
            continue
        found = True
        digest.update(name.encode("utf-8"))
        for key, tensor in sorted(child.state_dict().items()):
            digest.update(key.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest() if found else "unavailable"


def reference_from_firm_data(df: pd.DataFrame) -> Dict[str, float]:
    """Build one reference from parent rows without mutating the input frame."""
    source = df
    if "branch" in source.columns and bool((source["branch"] == -1).any()):
        source = source[source["branch"] == -1]
    aliases = {
        "x": ("x",),
        "i": ("i",),
        "Hatcf": ("Hatcf", "hatcf"),
        "LnKF": ("LnKF", "lnkf"),
    }
    result: Dict[str, float] = {}
    for canonical, candidates in aliases.items():
        column = next((name for name in candidates if name in source.columns), None)
        if column is None:
            raise ValueError(f"firm data is missing reference column {canonical!r}")
        values = pd.to_numeric(source[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            raise ValueError(f"firm data has no finite values for {column!r}")
        result[canonical] = float(finite.median())
    return result


def _surface_arrays(
    model: torch.nn.Module,
    reference: Mapping[str, float],
    *,
    device: torch.device,
    grid_size: int = 41,
) -> Dict[str, np.ndarray]:
    b_grid = torch.linspace(0.0, 1.0, grid_size, device=device)
    z_grid = torch.linspace(-4.0, 4.0, grid_size, device=device)
    bb, zz = torch.meshgrid(b_grid, z_grid, indexing="ij")
    result: Dict[str, np.ndarray] = {
        "b_grid": b_grid.cpu().numpy(),
        "z_grid": z_grid.cpu().numpy(),
        **{
            f"reference_{key}": np.asarray(float(value), dtype=np.float64)
            for key, value in reference.items()
        },
    }
    with torch.no_grad():
        for eta in (0, 1):
            state = torch.stack(
                [
                    bb.reshape(-1),
                    zz.reshape(-1),
                    torch.full_like(bb.reshape(-1), float(eta)),
                    torch.full_like(bb.reshape(-1), float(reference["i"])),
                    torch.full_like(bb.reshape(-1), float(reference["x"])),
                    torch.full_like(bb.reshape(-1), float(reference["Hatcf"])),
                    torch.full_like(bb.reshape(-1), float(reference["LnKF"])),
                ],
                dim=1,
            )
            output = model(state)
            phat = output.Phat
            q_claim = (
                model._q_claim_output(state)
                if getattr(model, "q_parameterization", None) == "hybrid_regime"
                else output.Q
            )
            q_unit = (
                model._q_unit_output(state)
                if getattr(model, "q_parameterization", None) == "hybrid_regime"
                else torch.where(
                    state[:, 0:1] > 0.0,
                    output.Q / state[:, 0:1],
                    torch.full_like(output.Q, float("nan")),
                )
            )
            recovery = model._q_recovery_output(state)
            values = {
                "P0": output.P0,
                "PI": output.PI,
                "Phat": phat,
                "P": output.P,
                "q_unit": q_unit,
                "Q_claim": q_claim,
                "Q_effective": output.Q,
                "recovery": recovery,
                "bp0": output.bp0,
                "bpI": output.bpI,
            }
            for key, value in values.items():
                result[f"{key}_eta{eta}"] = (
                    value.reshape(grid_size, grid_size).detach().cpu().numpy()
                )
    return result


def _sentinel_rows(
    model: torch.nn.Module,
    reference: Mapping[str, float],
    *,
    episode: int,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for eta in (0, 1):
                for b in (0.01, 0.05, 0.20, 0.50, 0.80):
                    for z in (-2.0, 0.0, 1.0, 2.0, 3.0, 4.0):
                        state = torch.tensor(
                            [[b, z, eta, reference["i"], reference["x"], reference["Hatcf"], reference["LnKF"]]],
                            dtype=next(model.parameters()).dtype,
                            device=device,
                        )
                        output = model(state)
                        q_claim = (
                            model._q_claim_output(state)
                            if getattr(model, "q_parameterization", None) == "hybrid_regime"
                            else output.Q
                        )
                        q_unit = (
                            model._q_unit_output(state)
                            if getattr(model, "q_parameterization", None) == "hybrid_regime"
                            else output.Q / state[:, 0:1]
                        )
                        rows.append({
                            "episode": episode,
                            "stage": "post_q_final",
                            "b": b,
                            "z": z,
                            "eta": eta,
                            "i": reference["i"],
                            "x": reference["x"],
                            "Hatcf": reference["Hatcf"],
                            "LnKF": reference["LnKF"],
                            "P0_pred": float(output.P0.item()),
                            "PI_pred": float(output.PI.item()),
                            "Phat_pred": float(output.Phat.item()),
                            "P_pred": float(output.P.item()),
                            "q_unit": float(q_unit.item()),
                            "Q_claim": float(q_claim.item()),
                            "Q_effective": float(output.Q.item()),
                            "recovery": float(model._q_recovery_output(state).item()),
                            "bp0": float(output.bp0.item()),
                            "bpI": float(output.bpI.item()),
                        })
    finally:
        model.train(was_training)
    return pd.DataFrame(rows)


def _finite_stats(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    result: Dict[str, float] = {}
    for source, prefix in (("P", "P"), ("Q", "Q"), ("b", "b"), ("bp", "bp"), ("z", "z"), ("M", "M")):
        if source not in df.columns:
            continue
        values = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        result[f"{prefix}_mean"] = float(np.mean(values))
        result[f"{prefix}_p90"] = float(np.quantile(values, 0.90))
        result[f"{prefix}_p99"] = float(np.quantile(values, 0.99))
    if "P" in df.columns:
        values = pd.to_numeric(df["P"], errors="coerce").to_numpy(dtype=float)
        result["default_rate"] = float(np.nanmean(values <= 0.0))
    if "Bar_i" in df.columns:
        values = pd.to_numeric(df["Bar_i"], errors="coerce").to_numpy(dtype=float)
        result["investment_rate"] = float(np.nanmean(values))
    return result


def _q_stage_summary(module_summary: Mapping[str, Any]) -> Dict[str, Any]:
    policy = module_summary.get("policy_value", {}) if isinstance(module_summary, Mapping) else {}
    metadata = policy.get("metadata", policy) if isinstance(policy, Mapping) else {}
    return metadata.get("q_regime_training_stage", {}) if isinstance(metadata, Mapping) else {}


def _phase(q_summary: Mapping[str, Any], name: str) -> Dict[str, Any]:
    for item in q_summary.get("phases", []):
        if item.get("phase") == name:
            return item
    return {}


def _stage_payload(
    *,
    episode: int,
    stage: str,
    hashes: Mapping[str, str],
    details: Mapping[str, Any],
) -> Dict[str, Any]:
    return {"episode": episode, "stage": stage, **hashes, **dict(details)}


def save_episode_diagnostics(
    *,
    run_root: Path,
    episode: int,
    model: torch.nn.Module,
    firm_df: pd.DataFrame,
    module_summary: Mapping[str, Any],
    canonical_reference: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Write diagnostics and return the unchanged run-global canonical reference."""
    root = Path(run_root) / "episode_diagnostics"
    episode_dir = root / f"ep_{episode:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    if canonical_reference is None:
        canonical_reference = reference_from_firm_data(firm_df)
        _write_json(root / "canonical_reference.json", canonical_reference)
    canonical_reference = {key: float(canonical_reference[key]) for key in REFERENCE_COLUMNS}
    ondist_reference = reference_from_firm_data(firm_df)

    hashes = {
        "model_hash": _state_hash(model),
        "p_hash": _module_hash(model, ("value_encoder", "v0_head", "vi_head", "barz_model")),
        "q_hash": _module_hash(model, ("q_encoder", "q_head")),
        "bp_hash": _module_hash(model, ("policy_encoder", "bp0_head", "bpi_head")),
    }
    q_summary = _q_stage_summary(module_summary)
    survival = _phase(q_summary, "survival")
    polish = _phase(q_summary, "polish")
    policy = module_summary.get("policy_value", {}) if isinstance(module_summary, Mapping) else {}
    policy_metadata = policy.get("metadata", policy) if isinstance(policy, Mapping) else {}
    p_summary = policy_metadata.get("policy_value_evaluation_stage", {}) if isinstance(policy_metadata, Mapping) else {}
    bp_summary = policy_metadata.get("bp_distillation_stage", {}) if isinstance(policy_metadata, Mapping) else {}
    stage_hashes = policy_metadata.get("stage_component_hashes", {}) if isinstance(policy_metadata, Mapping) else {}

    def hashes_for(stage: str) -> Mapping[str, str]:
        value = stage_hashes.get(stage) if isinstance(stage_hashes, Mapping) else None
        return value if isinstance(value, Mapping) else hashes

    _write_json(episode_dir / "00_episode_start.json", _stage_payload(
        episode=episode, stage="episode_start", hashes=hashes_for("episode_start"),
        details={"q_online_hash_stage_start": q_summary.get("q_online_hash_stage_start")},
    ))
    _write_json(episode_dir / "10_post_p.json", _stage_payload(
        episode=episode, stage="post_p", hashes=hashes_for("post_p"), details={"p": p_summary},
    ))
    _write_json(episode_dir / "20_post_q_survival.json", _stage_payload(
        episode=episode, stage="post_q_survival", hashes=hashes_for("post_q_survival"), details={"q": survival},
    ))
    _write_json(episode_dir / "30_post_q_final.json", _stage_payload(
        episode=episode, stage="post_q_final", hashes=hashes_for("post_q_final"), details={"q": q_summary},
    ))
    _write_json(episode_dir / "40_post_bp.json", _stage_payload(
        episode=episode, stage="post_bp", hashes=hashes_for("post_bp"), details={"bp": bp_summary},
    ))

    history = q_summary.get("q_validation_history", [])
    history_columns = (
        "episode", "phase", "epoch", "score_primary", "score_raw_abs",
        "score_normalized_abs", "q_unit_mean", "q_unit_p95", "q_unit_max",
        "q_claim_mean", "q_claim_p95", "q_claim_max", "recursion_gain_mean",
        "recursion_gain_p95", "recursion_gain_gt1_share", "lr", "target_hash",
        "online_q_hash", "is_best", "restored",
    )
    with (episode_dir / "q_validation_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_columns)
        writer.writeheader()
        for item in history:
            writer.writerow({
                "episode": episode,
                **{key: item.get(key) for key in history_columns if key not in {"episode", "q_claim_mean", "q_claim_p95", "q_claim_max", "recursion_gain_mean", "recursion_gain_p95", "recursion_gain_gt1_share"}},
                "q_claim_mean": item.get("q_claim_value_mean"),
                "q_claim_p95": item.get("q_claim_value_p95"),
                "q_claim_max": item.get("q_claim_value_max"),
                "recursion_gain_mean": item.get("q_recursion_gain_mean"),
                "recursion_gain_p95": item.get("q_recursion_gain_p95"),
                "recursion_gain_gt1_share": item.get("q_recursion_gain_gt1_share"),
            })

    _sentinel_rows(
        model, canonical_reference, episode=episode, device=next(model.parameters()).device
    ).to_csv(episode_dir / "sentinel_states.csv", index=False)
    was_training = model.training
    model.eval()
    try:
        np.savez_compressed(
            episode_dir / "surface_canonical.npz",
            **_surface_arrays(model, canonical_reference, device=next(model.parameters()).device),
        )
        np.savez_compressed(
            episode_dir / "surface_ondist.npz",
            **_surface_arrays(model, ondist_reference, device=next(model.parameters()).device),
        )
    finally:
        model.train(was_training)

    sim_stats = _finite_stats(firm_df)
    q_metrics = polish.get("metrics") or survival.get("metrics") or {}
    warnings = []
    if q_summary.get("status") == "accepted_reverted":
        warnings.append("q_validation_not_improved")
    if float(q_metrics.get("q_recursion_gain_gt1_share", 0.0) or 0.0) > 0.1:
        warnings.append("q_recursion_gain_gt1_share_large")
    health = {
        "episode": episode,
        "p": {"summary": p_summary},
        "q": {
            "status": q_summary.get("status"),
            "survival_status": survival.get("status"),
            "polish_status": polish.get("status"),
            "validation_start": survival.get("validation_start"),
            "validation_best": survival.get("validation_best"),
            "validation_final": polish.get("validation_after_restore") or survival.get("validation_after_restore"),
            "reverted": q_summary.get("status") == "accepted_reverted",
            "q_claim_p95": q_metrics.get("q_claim_value_p95"),
            "q_claim_p99": q_metrics.get("q_claim_value_p99"),
            "q_unit_p95": q_metrics.get("q_unit_p95"),
            "rho_q_p95": q_metrics.get("q_recursion_gain_p95"),
            "rho_q_gt1_share": q_metrics.get("q_recursion_gain_gt1_share"),
            "no_improvement_streak": q_summary.get("q_no_improvement_streak"),
        },
        "simulation": sim_stats,
        "warnings": warnings,
    }
    _write_json(episode_dir / "episode_health.json", health)
    return dict(canonical_reference)
