from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from analysis.convergence_transition import ConvergenceShockBank
from analysis.economic_config import AnalysisEconomicConfig
from evaluation.convergence_metrics import compute_shift_metrics, regression_metrics
from losses import SDFLoss
from losses.sdf_loss import compute_pooled_moment_constraints
from utils.metrics import conditional_moment_metrics


_CONFIG_ECONOMIC_FIELDS = (
    "RHO_X", "SIGMA_X", "XBAR", "ZETA", "G", "DELTA", "PHI", "TAU",
    "KAPPA_B", "KAPPA_E",
)


def _stable_config_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"Unsupported config invariant value: {type(value).__name__}")


def stable_config_hash(snapshot: Dict[str, Any]) -> str:
    """Hash semantic configuration only, excluding paths and runtime fields."""
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_config_invariant_snapshot(loaded: Any) -> tuple[Dict[str, Any], str]:
    """Build the cross-episode economic/model-semantic invariant snapshot."""
    economic = loaded.economic_config
    hp = loaded.hyperparams
    policy_model = loaded.models["policy_value"]
    sdf_fc1 = loaded.models["sdf_fc1"]
    sdf_model = sdf_fc1.sdf_model
    snapshot = {
        "economic": {
            name: _stable_config_value(getattr(economic, name))
            for name in _CONFIG_ECONOMIC_FIELDS
        },
        "sdf": {
            "wealth_residual_mode": str(getattr(hp, "sdf_wealth_loss_mode")),
            "normalized_logr_clip": float(getattr(hp, "sdf_normalized_logr_clip")),
            "gamma": float(sdf_model.gamma),
            "kappa": float(sdf_model.kappa),
            "sigma": float(sdf_model.sigma),
            "beta": float(sdf_model.beta),
        },
        "policy_value": {
            "pv_use_clipped_m": bool(getattr(hp, "pv_use_clipped_m")),
            "pv_m_clamp_min": float(getattr(hp, "pv_m_clamp_min")),
            "pv_m_clamp_max": float(getattr(hp, "pv_m_clamp_max")),
            "value_scale_mode": str(policy_model.value_scale_mode),
            "value_scale_log_max": float(policy_model.value_scale_log_max),
            "bellman_normalize_by_value_scale": bool(
                getattr(hp, "pv_bellman_normalize_by_value_scale")
            ),
            "eta_integration_semantics": (
                "exact_bernoulli"
                if bool(getattr(hp, "pv_exact_eta_integration_enabled"))
                else "sampled_eta"
            ),
        },
    }
    return snapshot, stable_config_hash(snapshot)


def _config_diff_fields(reference: Any, candidate: Any, *, prefix: str = "") -> list[str]:
    if isinstance(reference, dict) and isinstance(candidate, dict):
        fields: list[str] = []
        for key in sorted(set(reference) | set(candidate)):
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in reference or key not in candidate:
                fields.append(name)
            else:
                fields.extend(_config_diff_fields(reference[key], candidate[key], prefix=name))
        return fields
    return [] if reference == candidate else [prefix]


def compare_config_invariant_snapshots(
    records: Sequence[tuple[int, Dict[str, Any], str]],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Compare episode snapshots against the first available episode."""
    if not records:
        empty = pd.DataFrame(columns=[
            "episode", "config_hash", "comparable_to_reference", "diff_fields"
        ])
        return empty, {
            "reference_episode": None, "all_comparable": False,
            "mismatched_episodes": [], "fields": [],
        }
    ordered = sorted(records, key=lambda item: item[0])
    reference_episode, reference, reference_hash = ordered[0]
    rows = []
    all_fields: set[str] = set()
    mismatched: list[int] = []
    for episode, snapshot, config_hash in ordered:
        diff_fields = _config_diff_fields(reference, snapshot)
        all_fields.update(diff_fields)
        comparable = config_hash == reference_hash and not diff_fields
        if not comparable:
            mismatched.append(int(episode))
        rows.append({
            "episode": int(episode),
            "config_hash": config_hash,
            "comparable_to_reference": bool(comparable),
            "diff_fields": ";".join(diff_fields),
        })
    return pd.DataFrame(rows), {
        "reference_episode": int(reference_episode),
        "reference_config_hash": reference_hash,
        "all_comparable": not mismatched,
        "mismatched_episodes": mismatched,
        "fields": sorted(all_fields),
    }


def summarize_episode_statuses(
    frame: pd.DataFrame,
    *,
    n_requested: int | None = None,
) -> Dict[str, int]:
    counts = frame.get("status", pd.Series(dtype=object)).value_counts()
    result = {
        "n_requested": int(len(frame) if n_requested is None else n_requested),
        "n_ok": int(counts.get("ok", 0)),
        "n_partial": int(counts.get("partial", 0)),
        "n_error": int(counts.get("error", 0)),
        "n_missing": int(counts.get("missing", 0)),
    }
    result["n_not_fully_ok"] = (
        result["n_partial"] + result["n_error"] + result["n_missing"]
    )
    # Backward-compatible alias with corrected semantics.
    result["n_error_or_missing"] = result["n_error"] + result["n_missing"]
    return result


def model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _tensor_hash(tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _finite_residuals(residuals: torch.Tensor) -> torch.Tensor:
    values = residuals.detach().to(torch.float64)
    valid = torch.isfinite(values).all(dim=1)
    return values[valid]


def conditional_residual_summary(residuals: torch.Tensor, *, prefix: str) -> Dict[str, float]:
    values = _finite_residuals(residuals)
    if values.shape[0] == 0 or values.shape[1] < 2:
        return {f"{prefix}_{name}": float("nan") for name in (
            "conditional_mean_abs", "conditional_p50_abs", "conditional_p90_abs",
            "conditional_p95_abs", "conditional_p99_abs", "conditional_max_abs",
            "cm_mse", "u_stat", "raw_r_rms", "n_parents", "n_children",
        )}
    base = conditional_moment_metrics(values)
    absolute = values.mean(dim=1).abs()
    result = {
        "conditional_mean_abs": float(absolute.mean().item()),
        "conditional_p50_abs": float(torch.quantile(absolute, 0.50).item()),
        "conditional_p90_abs": float(torch.quantile(absolute, 0.90).item()),
        "conditional_p95_abs": float(torch.quantile(absolute, 0.95).item()),
        "conditional_p99_abs": float(torch.quantile(absolute, 0.99).item()),
        "conditional_max_abs": float(absolute.max().item()),
        "cm_mse": base["heldout_cm_mse"],
        "u_stat": base["heldout_u_stat"],
        "raw_r_rms": base["heldout_raw_r_rms"],
        "n_parents": int(values.shape[0]),
        "n_children": int(values.shape[1]),
    }
    return {f"{prefix}_{key}": value for key, value in result.items()}


def _summarize_sdf_prefix(
    *,
    raw_residual: torch.Tensor,
    normalized_residual: torch.Tensor,
    m_raw: torch.Tensor,
    log_r: torch.Tensor,
    loss: SDFLoss,
    normalized_logr_clip: float,
    n_parent: int,
    n_children: int,
) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    summary.update(conditional_residual_summary(raw_residual, prefix="sdf_raw"))
    summary.update(conditional_residual_summary(normalized_residual, prefix="sdf_normalized"))
    raw_finite = torch.isfinite(raw_residual)
    normalized_finite = torch.isfinite(normalized_residual)
    raw_valid_parent = raw_finite.all(dim=1)
    normalized_valid_parent = normalized_finite.all(dim=1)
    summary.update({
        "sdf_n_parents_requested": n_parent,
        "sdf_n_parents_valid": int(normalized_valid_parent.sum().item()),
        "sdf_valid_parent_ratio": float(normalized_valid_parent.double().mean().item()),
        "sdf_n_children": int(n_children),
        "sdf_finite_child_ratio": float(normalized_finite.double().mean().item()),
        "sdf_raw_valid_parent_ratio": float(raw_valid_parent.double().mean().item()),
        "sdf_normalized_valid_parent_ratio": float(
            normalized_valid_parent.double().mean().item()
        ),
        "sdf_raw_finite_child_ratio": float(raw_finite.double().mean().item()),
        "sdf_normalized_finite_child_ratio": float(
            normalized_finite.double().mean().item()
        ),
    })
    finite_m = m_raw.detach().reshape(-1).to(torch.float64)
    finite_m = finite_m[torch.isfinite(finite_m)]
    summary.update({
        "sdf_M_mean": float(finite_m.mean().item()) if finite_m.numel() else float("nan"),
        "sdf_M_std": float(finite_m.std(unbiased=False).item()) if finite_m.numel() else float("nan"),
        "sdf_M_min": float(finite_m.min().item()) if finite_m.numel() else float("nan"),
        "sdf_M_max": float(finite_m.max().item()) if finite_m.numel() else float("nan"),
        "sdf_M_finite_ratio": float(torch.isfinite(m_raw).double().mean().item()),
        "sdf_log_R_clip_share": float(
            (log_r.abs() > float(normalized_logr_clip)).double().mean().item()
        ),
    })
    if finite_m.numel():
        for quantile, label in (
            (0.01, "p01"), (0.05, "p05"), (0.50, "p50"),
            (0.95, "p95"), (0.99, "p99"),
        ):
            summary[f"sdf_M_{label}"] = float(torch.quantile(finite_m, quantile).item())
        constraints = compute_pooled_moment_constraints(
            finite_m, loss.mu_lo, loss.mu_hi, loss.var_hi
        )
        summary["sdf_g_max"] = float(constraints["max_violation"].item())
        for name in ("g_mean_low", "g_mean_high", "g_var_high"):
            summary[f"sdf_{name}"] = float(constraints[name].item())
    else:
        summary["sdf_g_max"] = float("nan")
        for name in ("g_mean_low", "g_mean_high", "g_var_high"):
            summary[f"sdf_{name}"] = float("nan")
        for label in ("p01", "p05", "p50", "p95", "p99"):
            summary[f"sdf_M_{label}"] = float("nan")
    summary.update({
        "heldout_mean_r_abs": summary["sdf_normalized_conditional_mean_abs"],
        "heldout_mean_r_p90": summary["sdf_normalized_conditional_p90_abs"],
        "heldout_mean_r_p99": summary["sdf_normalized_conditional_p99_abs"],
        "heldout_cm_mse": summary["sdf_normalized_cm_mse"],
        "heldout_u_stat": summary["sdf_normalized_u_stat"],
        "heldout_raw_r_rms": summary["sdf_raw_raw_r_rms"],
        "heldout_n_parents": summary["sdf_normalized_n_parents"],
        "heldout_n_children": summary["sdf_normalized_n_children"],
    })
    return summary


def evaluate_sdf_heldout_multi_k(
    model: torch.nn.Module,
    parent_states: torch.Tensor,
    *,
    hatc_cal: torch.Tensor,
    lnk_cal: torch.Tensor,
    economic_config: AnalysisEconomicConfig,
    child_counts: Sequence[int],
    seed: int,
    shock_bank_max_children: int | None = None,
    normalized_logr_clip: float = 20.0,
) -> tuple[Dict[int, Dict[str, float]], Dict[str, Any]]:
    """Evaluate nested child-count prefixes with one max-K SDF forward."""
    counts = sorted(set(int(value) for value in child_counts))
    if not counts or counts[0] < 2:
        raise ValueError("SDF held-out evaluation requires child counts >= 2")
    device, dtype = parent_states.device, parent_states.dtype
    n_parent = int(parent_states.shape[0])
    max_children = max(counts)
    bank_size = int(shock_bank_max_children or max_children)
    if bank_size < max_children:
        raise ValueError("shock_bank_max_children must be >= max(child_counts)")
    full_bank = ConvergenceShockBank.create(
        n_parent, bank_size, seed=seed, device=device, dtype=dtype
    )
    bank = ConvergenceShockBank(
        eps_x=full_bank.eps_x[:, :max_children],
        eps_z=full_bank.eps_z[:, :max_children],
        u_eta=full_bank.u_eta[:, :max_children],
        u_i=full_bank.u_i[:, :max_children],
        seed=full_bank.seed,
    )
    x_prev = parent_states[:, 4:5]
    x_next = (
        (1.0 - economic_config.RHO_X) * economic_config.XBAR
        + economic_config.RHO_X * x_prev.unsqueeze(1)
        + economic_config.SIGMA_X * bank.eps_x
    )
    with torch.inference_mode():
        w_parent, w_children, m_raw, hatc_next, lnk_next = model.forward_step(
            x_prev=x_prev,
            x_curr=x_next,
            hatcf_prev=hatc_cal,
            lnkf_prev=lnk_cal,
            return_physical=True,
        )
        loss = SDFLoss(
            gamma=float(model.sdf_model.gamma),
            kappa=float(model.sdf_model.kappa),
            sigma=float(model.sdf_model.sigma),
            beta=float(model.sdf_model.beta),
            wealth_loss_mode="signed_aio",
        )
        residuals = loss.compute_wealth_residuals(
            w_parent=w_parent,
            w_children=w_children,
            k_parent=lnk_cal,
            k_children=lnk_next,
            c_parent=hatc_cal,
            c_children=hatc_next,
            normalized_logr_clip=normalized_logr_clip,
        )
    summaries = {
        count: _summarize_sdf_prefix(
            raw_residual=residuals["raw"][:, :count],
            normalized_residual=residuals["normalized"][:, :count],
            m_raw=m_raw[:, :count],
            log_r=residuals["log_R"][:, :count],
            loss=loss,
            normalized_logr_clip=normalized_logr_clip,
            n_parent=n_parent,
            n_children=count,
        )
        for count in counts
    }
    metadata = {
        "source": "fresh_ConvergenceShockBank",
        "seed": int(seed),
        "n_parents": n_parent,
        "child_counts": counts,
        "max_children_evaluated": max_children,
        "shock_bank_max_children": bank_size,
        "sdf_forward_calls": 1,
        "common_random_nested_prefix_compatible": True,
        "parent_bank_sha256": _tensor_hash((
            ("parent_states", parent_states), ("hatc_cal", hatc_cal), ("lnk_cal", lnk_cal),
        )),
        "shock_bank_sha256": _tensor_hash((
            ("eps_x", full_bank.eps_x), ("eps_z", full_bank.eps_z),
            ("u_eta", full_bank.u_eta), ("u_i", full_bank.u_i),
        )),
        "residual_helper": "SDFLoss.compute_wealth_residuals",
        "conditional_metric_helper": "utils.metrics.conditional_moment_metrics",
    }
    return summaries, metadata


def evaluate_sdf_heldout(
    model: torch.nn.Module,
    parent_states: torch.Tensor,
    *,
    hatc_cal: torch.Tensor,
    lnk_cal: torch.Tensor,
    economic_config: AnalysisEconomicConfig,
    n_children: int,
    seed: int,
    shock_bank_max_children: int | None = None,
    normalized_logr_clip: float = 20.0,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """Backward-compatible single-K wrapper around the max-K evaluator."""
    summaries, metadata = evaluate_sdf_heldout_multi_k(
        model,
        parent_states,
        hatc_cal=hatc_cal,
        lnk_cal=lnk_cal,
        economic_config=economic_config,
        child_counts=[n_children],
        seed=seed,
        shock_bank_max_children=shock_bank_max_children,
        normalized_logr_clip=normalized_logr_clip,
    )
    return summaries[int(n_children)], {**metadata, "n_children": int(n_children)}


def namespace_sdf_summary(summary: Dict[str, float], *, scope: str) -> Dict[str, float]:
    """Give common/ondist SDF metrics explicit, non-overlapping names."""
    if scope not in {"common", "ondist"}:
        raise ValueError("scope must be 'common' or 'ondist'")
    prefix = f"sdf_{scope}"
    result: Dict[str, float] = {}
    metric_names = {
        "conditional_mean_abs": "conditional_abs_mean",
        "conditional_p50_abs": "conditional_abs_p50",
        "conditional_p90_abs": "conditional_abs_p90",
        "conditional_p95_abs": "conditional_abs_p95",
        "conditional_p99_abs": "conditional_abs_p99",
        "conditional_max_abs": "conditional_abs_max",
        "cm_mse": "cm_mse",
        "u_stat": "u_stat",
        "raw_r_rms": "raw_r_rms",
    }
    for residual_kind in ("raw", "normalized"):
        for name, destination_name in metric_names.items():
            source = f"sdf_{residual_kind}_{name}"
            result[f"{prefix}_{residual_kind}_{destination_name}"] = summary.get(source, float("nan"))
    # The unqualified scope metrics are normalized residuals and are the primary
    # cross-episode SDF convergence semantics.
    for name, destination_name in metric_names.items():
        source = f"sdf_normalized_{name}"
        result[f"{prefix}_{destination_name}"] = summary.get(source, float("nan"))
    direct = (
        "n_parents_requested", "n_parents_valid", "valid_parent_ratio",
        "n_children", "finite_child_ratio", "M_mean", "M_std", "M_min", "M_max",
        "M_p01", "M_p05", "M_p50", "M_p95", "M_p99", "M_finite_ratio",
        "log_R_clip_share", "g_mean_low", "g_mean_high", "g_var_high", "g_max",
        "raw_valid_parent_ratio", "normalized_valid_parent_ratio",
        "raw_finite_child_ratio", "normalized_finite_child_ratio",
    )
    for name in direct:
        result[f"{prefix}_{name}"] = summary.get(f"sdf_{name}", float("nan"))
    return result


def _normalize_macro(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "Hatc": "hatc_cal", "Hatc_cal": "hatc_cal", "HATC": "hatc_cal",
        "LnK": "lnk_cal", "LnK_cal": "lnk_cal", "LNK": "lnk_cal",
        "hatcf": "Hatcf", "lnkf": "LnKF", "X": "x",
    }
    return frame.rename(columns={k: v for k, v in aliases.items() if k in frame and v not in frame})


def _one_step_frame(model: torch.nn.Module, frame: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    data = _normalize_macro(frame).copy()
    required = {"path", "t", "x", "hatc_cal", "lnk_cal"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"macro dataframe missing FC1 columns: {missing}")
    data["t"] = pd.to_numeric(data["t"], errors="coerce")
    data = data.sort_values(["path", "t"]).drop_duplicates(["path", "t"])
    nxt = data[["path", "t", "x", "hatc_cal", "lnk_cal"]].copy()
    nxt["t"] -= 1
    nxt = nxt.rename(columns={
        "x": "x_next", "hatc_cal": "hatc_target", "lnk_cal": "lnk_target"
    })
    paired = data.merge(nxt, on=["path", "t"], how="inner")
    numeric = paired[["x", "x_next", "hatc_cal", "lnk_cal", "hatc_target", "lnk_target"]]
    mask = np.isfinite(numeric.apply(pd.to_numeric, errors="coerce").to_numpy()).all(axis=1)
    paired = paired.loc[mask].copy()
    if paired.empty:
        raise ValueError("macro dataframe has no aligned finite FC1 one-step pairs")
    tensor = lambda name: torch.as_tensor(
        paired[name].to_numpy(dtype=np.float32), device=device
    ).reshape(-1, 1)
    with torch.no_grad():
        _, _, _, hatc_pred, lnk_pred = model.forward_step(
            tensor("x"), tensor("x_next"), tensor("hatc_cal"), tensor("lnk_cal")
        )
    # Evaluator-generated forecasts always use dedicated ``*_eval`` names so we
    # can never collide with (or silently duplicate) artifact columns such as
    # Hatcf / LnKF that may already be present in the macro frame.
    for column in ("hatc_pred_eval", "lnk_pred_eval"):
        if column in paired.columns:
            paired = paired.drop(columns=column)
    paired["hatc_pred_eval"] = hatc_pred.detach().cpu().reshape(-1).numpy()
    paired["lnk_pred_eval"] = lnk_pred.detach().cpu().reshape(-1).numpy()
    return paired


def evaluate_fc1_checkpoint(
    model: torch.nn.Module,
    macro_frame: pd.DataFrame,
    *,
    device: torch.device,
    rollout_horizons: Sequence[int] = (1, 5, 10, 20),
    shifts: Sequence[int] = (-2, -1, 0, 1, 2),
) -> tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    paired = _one_step_frame(model, macro_frame, device)
    summary: Dict[str, float] = {}
    for name in ("hatc", "lnk"):
        forecast_col = f"{name}_pred_eval"
        target_col = f"{name}_target"
        metrics = regression_metrics(
            paired[target_col],
            paired[forecast_col],
            calculated_name=target_col,
            forecast_name=forecast_col,
        )
        summary.update({f"fc1_{name}_{key}": value for key, value in metrics.items()})
        persistence_rmse = float(np.sqrt(np.mean(np.square(
            paired[target_col].to_numpy() - paired[f"{name}_cal"].to_numpy()
        ))))
        summary[f"fc1_{name}_persistence_rmse"] = persistence_rmse
        summary[f"fc1_{name}_persistence_skill"] = (
            1.0 - float(metrics["rmse"]) / persistence_rmse
            if np.isfinite(persistence_rmse) and persistence_rmse > 0 else float("nan")
        )

    timing_rows = []
    for name in ("hatc", "lnk"):
        forecast_col = f"{name}_pred_eval"
        target_col = f"{name}_target"
        # Targets in paired are already t+1. Treat that alignment as shift 0.
        base = paired[["path", "t", forecast_col, target_col]].copy()
        for shift in shifts:
            target = base[["path", "t", target_col]].copy()
            target["t"] = target["t"] - int(shift)
            target = target.rename(columns={target_col: "target_shifted"})
            aligned = base.merge(target, on=["path", "t"], how="inner")
            metrics = regression_metrics(
                aligned["target_shifted"],
                aligned[forecast_col],
                calculated_name="target_shifted",
                forecast_name=forecast_col,
            )
            timing_rows.append({"variable": name, "shift": int(shift), **metrics})
    timing = pd.DataFrame(timing_rows)
    for name in ("hatc", "lnk"):
        subset = timing[timing["variable"] == name]
        finite = subset[np.isfinite(subset["rmse"])]
        summary[f"fc1_{name}_best_timing_shift"] = (
            int(finite.loc[finite["rmse"].idxmin(), "shift"]) if not finite.empty else float("nan")
        )

    rollout_rows = []
    source = _normalize_macro(macro_frame).sort_values(["path", "t"])
    for horizon in rollout_horizons:
        finite_count = total_count = 0
        hatc_errors: list[float] = []
        lnk_errors: list[float] = []
        for _, group in source.groupby("path"):
            group = group.reset_index(drop=True)
            for start in range(max(0, len(group) - int(horizon))):
                current_hatc = torch.tensor([[float(group.loc[start, "hatc_cal"])]], device=device)
                current_lnk = torch.tensor([[float(group.loc[start, "lnk_cal"])]], device=device)
                ok = True
                with torch.no_grad():
                    for step in range(int(horizon)):
                        x0 = torch.tensor([[float(group.loc[start + step, "x"])]], device=device)
                        x1 = torch.tensor([[float(group.loc[start + step + 1, "x"])]], device=device)
                        _, _, _, current_hatc, current_lnk = model.forward_step(
                            x0, x1, current_hatc, current_lnk
                        )
                        ok = ok and bool(torch.isfinite(current_hatc).all() and torch.isfinite(current_lnk).all())
                total_count += 1
                finite_count += int(ok)
                if ok:
                    hatc_errors.append(
                        float(current_hatc.item()) - float(group.loc[start + int(horizon), "hatc_cal"])
                    )
                    lnk_errors.append(
                        float(current_lnk.item()) - float(group.loc[start + int(horizon), "lnk_cal"])
                    )
        ratio = float(finite_count / total_count) if total_count else float("nan")
        hatc_rmse = float(np.sqrt(np.mean(np.square(hatc_errors)))) if hatc_errors else float("nan")
        lnk_rmse = float(np.sqrt(np.mean(np.square(lnk_errors)))) if lnk_errors else float("nan")
        hatc_mae = float(np.mean(np.abs(hatc_errors))) if hatc_errors else float("nan")
        lnk_mae = float(np.mean(np.abs(lnk_errors))) if lnk_errors else float("nan")
        rollout_rows.append({
            "horizon": int(horizon), "finite_ratio": ratio, "n_origins": total_count,
            "hatc_rmse": hatc_rmse, "hatc_mae": hatc_mae,
            "lnk_rmse": lnk_rmse, "lnk_mae": lnk_mae,
        })
        summary[f"fc1_rollout_h{int(horizon)}_finite_ratio"] = ratio
        summary[f"fc1_rollout_h{int(horizon)}_hatc_rmse"] = hatc_rmse
        summary[f"fc1_rollout_h{int(horizon)}_hatc_mae"] = hatc_mae
        summary[f"fc1_rollout_h{int(horizon)}_lnk_rmse"] = lnk_rmse
        summary[f"fc1_rollout_h{int(horizon)}_lnk_mae"] = lnk_mae
    return summary, timing, pd.DataFrame(rollout_rows)


_EPISODE_RE = re.compile(r"Episode\s+(?P<episode>\d+)", re.IGNORECASE)
_SDF_ARROW_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _sdf_stage_from_line(line: str, current_stage: str | None) -> str:
    lowered = line.lower().replace("-", "_")
    if "sdf_true" in lowered or "sdf true" in lowered:
        return "sdf_true_only"
    if "sdf_recursive" in lowered:
        return "sdf_recursive_only"
    if "ep0_sdf" in lowered or "sdf bootstrap" in lowered:
        return "episode0_bootstrap"
    return current_stage or "unknown"


def parse_sdf_validation_log_blocks(path: str | Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Parse every SDF validation block without silently collapsing repeats."""
    path = Path(path)
    current_episode: int | None = None
    current_stage: str | None = None
    occurrences: Dict[int, int] = {}
    rows: list[Dict[str, Any]] = []
    arrow_patterns = {
        "normalized_mean": re.compile(
            rf"normalized_mean\s*=\s*({_SDF_ARROW_NUMBER})\s*->\s*({_SDF_ARROW_NUMBER})",
            re.IGNORECASE,
        ),
        "normalized_t": re.compile(
            rf"normalized_t\s*=\s*({_SDF_ARROW_NUMBER})\s*->\s*({_SDF_ARROW_NUMBER})",
            re.IGNORECASE,
        ),
        "constraint": re.compile(
            rf"max_constraint_violation\s*=\s*({_SDF_ARROW_NUMBER})\s*->\s*({_SDF_ARROW_NUMBER})",
            re.IGNORECASE,
        ),
    }
    bool_pattern = re.compile(
        r"(?P<key>safe_to_continue|stage_progress|converged)\s*=\s*(?P<value>true|false)",
        re.IGNORECASE,
    )
    explicit_stage_pattern = re.compile(r"\bstage\s*=\s*(?P<stage>[A-Za-z0-9_.-]+)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        episode_match = _EPISODE_RE.search(line)
        if episode_match:
            current_episode = int(episode_match.group("episode"))
            current_stage = None
        inferred_stage = _sdf_stage_from_line(line, current_stage)
        current_stage = inferred_stage
        if current_episode is None or "SDF validation" not in line:
            continue
        explicit_stage = explicit_stage_pattern.search(line)
        if explicit_stage is not None:
            block_stage = explicit_stage.group("stage").lower().replace("-", "_")
            stage_source = "explicit"
        else:
            block_stage = inferred_stage
            stage_source = "inferred"
        occurrence = occurrences.get(current_episode, 0) + 1
        occurrences[current_episode] = occurrence
        row: Dict[str, Any] = {
            "episode": current_episode,
            "stage": block_stage or "unknown",
            "stage_source": stage_source,
            "occurrence": occurrence,
        }
        for match in bool_pattern.finditer(line):
            row[match.group("key").lower()] = match.group("value").lower() == "true"
        for name, pattern in arrow_patterns.items():
            match = pattern.search(line)
            row[f"{name}_before"] = float(match.group(1)) if match else float("nan")
            row[f"{name}_after"] = float(match.group(2)) if match else float("nan")
        result_match = re.search(r"\bresult\s*=\s*(.+?)\s*$", line, re.IGNORECASE)
        row["result"] = result_match.group(1).strip() if result_match else ""
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame, {
        "source": "explicit_sdf_validation_blocks",
        "path": str(path.resolve()),
        "n_blocks": len(rows),
        "primary_selection_rule": (
            "exactly one sdf_true_only block per episode; otherwise headline unavailable"
        ),
    }


def select_primary_sdf_validation_blocks(
    blocks: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Select an unambiguous formal true-SDF validation block per episode."""
    if blocks.empty:
        return pd.DataFrame(columns=["episode"]), {"ambiguous_episodes": [], "missing_episodes": []}
    rows: list[Dict[str, Any]] = []
    ambiguous: list[int] = []
    for episode, group in blocks.groupby("episode", sort=True):
        formal = group[group["stage"] == "sdf_true_only"]
        explicit_formal = (
            formal[formal["stage_source"] == "explicit"]
            if "stage_source" in formal.columns else formal.iloc[0:0]
        )
        candidates = explicit_formal if not explicit_formal.empty else formal
        if len(candidates) != 1:
            ambiguous.append(int(episode))
            continue
        item = candidates.iloc[0]
        rows.append({
            "episode": int(episode),
            "sdf_before_aio_mean": item.get("normalized_mean_before", np.nan),
            "sdf_after_aio_mean": item.get("normalized_mean_after", np.nan),
            "sdf_before_aio_t": item.get("normalized_t_before", np.nan),
            "sdf_after_aio_t": item.get("normalized_t_after", np.nan),
            "sdf_constraint_before": item.get("constraint_before", np.nan),
            "sdf_constraint_after": item.get("constraint_after", np.nan),
            "sdf_safe_to_continue": item.get("safe_to_continue", np.nan),
            "sdf_stage_progress": item.get("stage_progress", np.nan),
            "sdf_converged": item.get("converged", np.nan),
            "sdf_validation_result": item.get("result", ""),
            "sdf_validation_stage": item.get("stage", ""),
            "sdf_validation_stage_source": item.get("stage_source", "inferred"),
            "sdf_validation_occurrence": item.get("occurrence", np.nan),
        })
    return pd.DataFrame(rows), {
        "ambiguous_episodes": ambiguous,
        "selection_rule": "exactly_one_explicit_sdf_true_only_else_exactly_one_inferred",
    }


def parse_training_log(path: str | Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Parse only explicit key=value diagnostics; unknown lines are preserved as unavailable."""
    path = Path(path)
    rows: Dict[int, Dict[str, Any]] = {}
    current: int | None = None
    key_value = re.compile(
        r"(?P<key>[A-Za-z][A-Za-z0-9_./-]*)\s*=\s*"
        r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|true|false|nan|[-+]?inf)",
        re.IGNORECASE,
    )
    nonfinite_values = 0
    repeated_keys = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _EPISODE_RE.search(line)
        if match:
            current = int(match.group("episode"))
            rows.setdefault(current, {"episode": current})
        if current is None:
            continue
        for item in key_value.finditer(line):
            key = item.group("key").replace("/", "_").replace(".", "_")
            raw_value = item.group("value")
            if key in rows[current]:
                repeated_keys += 1
            if raw_value.lower() in {"true", "false"}:
                rows[current][key] = raw_value.lower() == "true"
                continue
            value = float(raw_value)
            if not math.isfinite(value):
                nonfinite_values += 1
                rows[current][key] = float("nan")
            else:
                rows[current][key] = value
    return pd.DataFrame(list(rows.values())).sort_values("episode") if rows else pd.DataFrame(), {
        "source": "explicit_training_log",
        "path": str(path.resolve()),
        "parser": "episode_header_plus_scalar_key_value_last_occurrence",
        "repeated_key_occurrences": repeated_keys,
        "nonfinite_values_recorded_as_nan": nonfinite_values,
    }


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
