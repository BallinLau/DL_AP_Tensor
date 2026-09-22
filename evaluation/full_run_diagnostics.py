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
from losses import FC2Loss, SDFLoss
from losses.sdf_loss import compute_pooled_moment_constraints
from utils.metrics import conditional_moment_metrics


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
    """Fresh-shock held-out SDF/wealth residual evaluation."""
    if n_children < 2:
        raise ValueError("SDF held-out evaluation requires at least two children")
    device, dtype = parent_states.device, parent_states.dtype
    n_parent = int(parent_states.shape[0])
    bank_size = int(shock_bank_max_children or n_children)
    if bank_size < n_children:
        raise ValueError("shock_bank_max_children must be >= n_children")
    full_bank = ConvergenceShockBank.create(
        n_parent, bank_size, seed=seed, device=device, dtype=dtype
    )
    bank = ConvergenceShockBank(
        eps_x=full_bank.eps_x[:, :n_children],
        eps_z=full_bank.eps_z[:, :n_children],
        u_eta=full_bank.u_eta[:, :n_children],
        u_i=full_bank.u_i[:, :n_children],
        seed=full_bank.seed,
    )
    x_prev = parent_states[:, 4:5]
    x_next = (
        (1.0 - economic_config.RHO_X) * economic_config.XBAR
        + economic_config.RHO_X * x_prev.unsqueeze(1)
        + economic_config.SIGMA_X * bank.eps_x
    )
    with torch.no_grad():
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
    summary: Dict[str, float] = {}
    summary.update(conditional_residual_summary(residuals["raw"], prefix="sdf_raw"))
    summary.update(conditional_residual_summary(residuals["normalized"], prefix="sdf_normalized"))
    finite_m = m_raw.detach().reshape(-1).to(torch.float64)
    finite_m = finite_m[torch.isfinite(finite_m)]
    summary.update({
        "sdf_M_mean": float(finite_m.mean().item()) if finite_m.numel() else float("nan"),
        "sdf_M_std": float(finite_m.std(unbiased=False).item()) if finite_m.numel() else float("nan"),
        "sdf_M_min": float(finite_m.min().item()) if finite_m.numel() else float("nan"),
        "sdf_M_max": float(finite_m.max().item()) if finite_m.numel() else float("nan"),
        "sdf_M_finite_ratio": float(torch.isfinite(m_raw).double().mean().item()),
        "sdf_log_R_clip_share": float(residuals["log_R_clip_share"].item()),
    })
    if finite_m.numel():
        for quantile, label in ((0.01, "p01"), (0.05, "p05"), (0.50, "p50"), (0.95, "p95"), (0.99, "p99")):
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
    metadata = {
        "source": "fresh_ConvergenceShockBank",
        "seed": int(seed),
        "n_parents": n_parent,
        "n_children": int(n_children),
        "shock_bank_max_children": bank_size,
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
    return summary, metadata


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
    paired["hatc_pred"] = hatc_pred.detach().cpu().reshape(-1).numpy()
    paired["lnk_pred"] = lnk_pred.detach().cpu().reshape(-1).numpy()
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
        metrics = regression_metrics(paired[f"{name}_target"], paired[f"{name}_pred"])
        summary.update({f"fc1_{name}_{key}": value for key, value in metrics.items()})
        persistence_rmse = float(np.sqrt(np.mean(np.square(
            paired[f"{name}_target"].to_numpy() - paired[f"{name}_cal"].to_numpy()
        ))))
        summary[f"fc1_{name}_persistence_rmse"] = persistence_rmse
        summary[f"fc1_{name}_persistence_skill"] = (
            1.0 - float(metrics["rmse"]) / persistence_rmse
            if np.isfinite(persistence_rmse) and persistence_rmse > 0 else float("nan")
        )

    timing_rows = []
    timing_source = paired.rename(columns={"hatc_pred": "Hatcf", "lnk_pred": "LnKF"})
    for name, forecast, calculated in (
        ("hatc", "Hatcf", "hatc_target"), ("lnk", "LnKF", "lnk_target")
    ):
        # Targets in paired are already t+1. Treat that alignment as shift 0.
        base = timing_source[["path", "t", forecast, calculated]].copy()
        for shift in shifts:
            target = base[["path", "t", calculated]].copy()
            target["t"] -= int(shift)
            target = target.rename(columns={calculated: "target_shifted"})
            aligned = base.merge(target, on=["path", "t"], how="inner")
            metrics = regression_metrics(aligned["target_shifted"], aligned[forecast])
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


def evaluate_fc2_checkpoint(
    fc2_model: torch.nn.Module,
    policy_model: torch.nn.Module,
    firm_frame: pd.DataFrame,
    macro_frame: pd.DataFrame,
    *,
    device: torch.device,
    economic_config: AnalysisEconomicConfig,
) -> tuple[Dict[str, float], pd.DataFrame]:
    """Evaluate current FC2 parent/child aggregation semantics without training."""
    if fc2_model is None:
        return {}, pd.DataFrame()
    firm = firm_frame.copy()
    _ = macro_frame  # Kept in the interface because availability is checked by the orchestrator.
    required = {"b", "z", "x", "ETA", "i"}
    if "path" not in firm or not required.issubset(firm.columns):
        raise ValueError("FC2 evaluation requires path and five firm-state columns")
    if "branch" not in firm:
        firm["branch"] = 0
    branch_numeric = pd.to_numeric(firm["branch"], errors="coerce")
    parent_branch = -1 if bool((branch_numeric < 0).any()) else 0
    keys = ["path"] + (["t"] if "t" in firm else []) + ["branch"]
    rows = []
    quantiles = torch.linspace(0, 1, steps=int(fc2_model.quantile_num), device=device)
    loss = FC2Loss(delta=economic_config.DELTA, phi=economic_config.PHI)
    with torch.no_grad():
        for group_key, group in firm.groupby(keys):
            ordered = ["b", "z", "ETA", "i", "x"]
            numeric = group[ordered + (["K"] if "K" in group else [])].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            if numeric.empty:
                continue
            phi = torch.cat([
                torch.quantile(torch.as_tensor(numeric["b"].to_numpy(), device=device, dtype=torch.float32), quantiles),
                torch.quantile(torch.as_tensor(numeric["z"].to_numpy(), device=device, dtype=torch.float32), quantiles),
                torch.tensor([float(numeric["x"].mean())], device=device),
            ]).unsqueeze(0)
            pred = fc2_model(phi)
            key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
            base_states = torch.as_tensor(numeric[ordered].to_numpy(np.float32), device=device)
            # Match FC2LossPipe exactly: it concatenates [lnk_pred, hatc_pred]
            # after the five firm states before calling PolicyValueModel.
            states = torch.cat([
                base_states,
                pred["lnk"].expand(len(numeric), 1),
                pred["hatc"].expand(len(numeric), 1),
            ], dim=1)
            policy = policy_model(states)
            get = lambda name: policy[name] if isinstance(policy, dict) else getattr(policy, name)
            k = torch.as_tensor(
                numeric["K"].to_numpy(np.float32) if "K" in numeric else np.ones(len(numeric), np.float32),
                device=device,
            )
            y, investment, adjustment, consumption = loss.compute_resource_accounting(
                k, states[:, 1], states[:, 4], get("bar_i").reshape(-1),
                get("bar_z").reshape(-1), states[:, 3],
            )
            # FC2LossPipe trains against absolute consumption and treats any
            # positive bar_z as present in the aggregate. Preserve that actual
            # checkpoint target here and report its resource-accounting effect.
            consumption_used = consumption.abs()
            aggregate_mask = (get("bar_z").reshape(-1) > 0).to(k.dtype)
            hatc_agg, lnk_agg = loss.aggregate(k, consumption_used, aggregate_mask)
            resource_residual = y - consumption_used - investment - adjustment
            branch_value = float(pd.to_numeric(group["branch"], errors="coerce").iloc[0])
            rows.append({
                **dict(zip(keys, key_tuple)),
                "node_role": "parent" if branch_value == parent_branch else "child",
                "hatc_fc2": float(pred["hatc"].item()),
                "lnk_fc2": float(pred["lnk"].item()),
                "hatc_agg": float(hatc_agg.item()),
                "lnk_agg": float(lnk_agg.item()),
                "resource_residual_mean": float(resource_residual.mean().item()),
                "resource_residual_abs_max": float(resource_residual.abs().max().item()),
                "resource_output_abs_mean": float(y.abs().mean().item()),
                "consumption_finite_ratio": float(torch.isfinite(consumption).float().mean().item()),
            })
    frame = pd.DataFrame(rows)
    summary: Dict[str, float] = {}
    if not frame.empty:
        for name in ("hatc", "lnk"):
            err = frame[f"{name}_fc2"] - frame[f"{name}_agg"]
            absolute = np.abs(err.to_numpy(dtype=np.float64))
            summary[f"fc2_{name}_rmse"] = float(np.sqrt(np.mean(np.square(err))))
            summary[f"fc2_{name}_mae"] = float(absolute.mean())
            summary[f"fc2_{name}_p90"] = float(np.quantile(absolute, 0.90))
            summary[f"fc2_{name}_p99"] = float(np.quantile(absolute, 0.99))
            summary[f"fc2_{name}_max"] = float(absolute.max())
        summary["fc2_n_nodes"] = int(len(frame))
        summary["fc2_resource_residual_abs_mean"] = float(
            frame["resource_residual_mean"].abs().mean()
        )
        summary["fc2_resource_residual_abs_max"] = float(frame["resource_residual_abs_max"].max())
        resource_scale = np.maximum(frame["resource_output_abs_mean"].to_numpy(dtype=np.float64), 1e-8)
        summary["fc2_resource_residual_relative_abs_mean"] = float(
            np.mean(np.abs(frame["resource_residual_mean"].to_numpy(dtype=np.float64)) / resource_scale)
        )
        summary["fc2_consumption_finite_ratio"] = float(frame["consumption_finite_ratio"].mean())
        child = frame[frame["node_role"] == "child"]
        summary["fc2_transition_consistency_available"] = bool(not child.empty)
        for name in ("hatc", "lnk"):
            child_error = (
                child[f"{name}_fc2"].to_numpy(dtype=np.float64)
                - child[f"{name}_agg"].to_numpy(dtype=np.float64)
            )
            summary[f"fc2_transition_{name}_rmse"] = (
                float(np.sqrt(np.mean(np.square(child_error)))) if child_error.size else float("nan")
            )
            summary[f"fc2_transition_{name}_mae"] = (
                float(np.mean(np.abs(child_error))) if child_error.size else float("nan")
            )
    return summary, frame


_EPISODE_RE = re.compile(r"Episode\s+(?P<episode>\d+)", re.IGNORECASE)


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
