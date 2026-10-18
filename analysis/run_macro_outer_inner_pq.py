#!/usr/bin/env python3
"""SDF+FC1 macro outer around the profiled P/Q firm fixed-point solver.

The experiment is deliberately analysis-only.  FC2 is excluded structurally:
the simulator receives no FC2 module, the macro optimizer contains only the
combined SDF/FC1 module, and an installed FC2 forward guard raises on access.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from types import MethodType
from typing import Any, Callable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.run_coupled_fixed_macro import (  # noqa: E402
    _clone_state,
    build_coupled_view,
    coupled_transition_metrics,
    plot_level_maps,
    solve_p_block,
    solve_q_block,
)
from analysis.run_q_root_solve_fixed_macro import (  # noqa: E402
    _load_setup,
    build_q_survival_problem,
    evaluate_q_problem,
)
from analysis.test_frozen_policy_root_solve import (  # noqa: E402
    _tensor_digest,
    build_problem,
    evaluate_problem,
)
from data.simulate_ts import SimulateTS  # noqa: E402
from data.simulate_ts_parallel import _initialize_batched_state  # noqa: E402
from data.state_bank import clone_state  # noqa: E402
from experiments.run_utils import build_optimizers  # noqa: E402


DEFAULT_OUTPUT = ROOT / "cachedir" / "macro_outer_inner_pq_20261017"
DEFAULT_REPORT = ROOT / "reports" / "macro_outer_inner_pq_20261017.md"


def module_hash(module: Optional[torch.nn.Module]) -> str:
    if module is None:
        return "ABSENT"
    return _tensor_digest(sorted(module.state_dict().items()))


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return _tensor_digest(sorted((name, value) for name, value in state.items()))


def stable_json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def set_stage_trainability(models: Mapping[str, torch.nn.Module], stage: str) -> list[str]:
    """Set the only legal trainable module for an outer stage."""
    if stage not in {"macro", "firm", "frozen"}:
        raise ValueError(f"unknown stage {stage!r}")
    trainable = []
    for name, model in models.items():
        allow = (stage == "macro" and name == "sdf_fc1") or (
            stage == "firm" and name == "policy_value"
        )
        model.requires_grad_(allow)
        model.train(allow)
        if allow:
            trainable.append(name)
    return trainable


def install_fc2_guard(fc2: Optional[torch.nn.Module], counter: dict[str, int]) -> None:
    if fc2 is None:
        return

    def forbidden_forward(_self, *_args, **_kwargs):
        counter["calls"] = counter.get("calls", 0) + 1
        raise RuntimeError("FC2 MUST NOT BE USED IN MACRO OUTER")

    fc2.forward = MethodType(forbidden_forward, fc2)


def run_macro_outer_skeleton(
    state: Any,
    outer_max: int,
    firm_solve: Callable[[Any, int], Any],
    simulate: Callable[[Any, int], Any],
    macro_update: Callable[[Any, Any, int], Any],
) -> list[Any]:
    """Small dependency-injection skeleton used by invariant tests."""
    if not 0 <= outer_max <= 5:
        raise ValueError("macro outer max must be in [0, 5]")
    states = []
    current = state
    for outer in range(outer_max + 1):
        current = firm_solve(current, outer)
        data = simulate(current, outer)
        states.append(current)
        if outer < outer_max:
            current = macro_update(current, data, outer)
    return states


def refresh_transition_bank(
    sdf_fc1: torch.nn.Module,
    parents: torch.Tensor,
    children: torch.Tensor,
    *,
    m_min: float,
    m_max: float,
    preserve_child_macro: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Refresh Macro values on fixed shocks.

    Market-cleared full-economy banks already contain realized child clearing
    roots and calculated capital.  Those columns must remain fixed while the
    current SDF M is refreshed.  Legacy shock-only banks continue to use the
    current FC1 forecasts for their child macro columns.
    """
    if parents.shape[-1] < 7 or children.shape[-1] < 7:
        raise ValueError("transition bank must contain the seven firm state columns")
    if not m_min < m_max:
        raise ValueError("invalid common M clamp")
    with torch.no_grad():
        _, _, m_raw, hatc, lnk = sdf_fc1.forward_step(
            x_prev=parents[:, 4:5],
            x_curr=children[:, :, 4:5],
            hatcf_prev=parents[:, 5:6],
            lnkf_prev=parents[:, 6:7],
            return_physical=True,
        )
        m = m_raw.clamp(m_min, m_max)
        width = max(int(children.shape[-1]), 8)
        refreshed = torch.zeros(
            (*children.shape[:-1], width), device=children.device, dtype=children.dtype
        )
        refreshed[..., : children.shape[-1]] = children
        if not preserve_child_macro:
            refreshed[..., 5:6] = hatc
            refreshed[..., 6:7] = lnk
        refreshed[..., 7:8] = m
    flat = m.detach().reshape(-1).to(torch.float64)
    return refreshed, {
        "m_source": "CURRENT_SDF_FC1_CHILD_CLAMPED_ONCE",
        "m_hash": _tensor_digest([("m", m)]),
        "m_mean": float(flat.mean()),
        "m_p10": float(torch.quantile(flat, 0.10)),
        "m_p50": float(torch.quantile(flat, 0.50)),
        "m_p90": float(torch.quantile(flat, 0.90)),
        "m_raw_clip_share": float((m_raw != m).to(torch.float32).mean()),
        "macro_input_hash": _tensor_digest([
            ("hatc", hatc), ("lnk", lnk), ("m", m)
        ]),
        "child_macro_source": (
            "REALIZED_MARKET_CLEARING_ROOT_AND_CALCULATED_CAPITAL"
            if preserve_child_macro
            else "CURRENT_FC1_FORECAST"
        ),
        "child_hatc_prediction_mae": float(
            (hatc - children[..., 5:6]).abs().mean()
        ),
        "child_lnk_prediction_mae": float(
            (lnk - children[..., 6:7]).abs().mean()
        ),
    }


def expanded_operator_m(
    episode: Any, children: torch.Tensor, *, source: str
) -> torch.Tensor:
    child_list = list(children.unbind(dim=1))
    if any(child.shape[-1] <= 7 for child in child_list):
        raise ValueError("operator M audit requires child column 7")
    if source not in {"p", "q"}:
        raise ValueError(source)
    common = [child[:, 7:8].detach() for child in child_list]
    _, _, expanded, _ = episode._expand_policy_expectation_children(
        child_list, common, common
    )
    return torch.stack(expanded, dim=1)


def audit_m_parity(episode: Any, children: torch.Tensor) -> dict[str, Any]:
    p_m = expanded_operator_m(episode, children, source="p")
    q_m = expanded_operator_m(episode, children, source="q")
    difference = float((p_m - q_m).abs().max())
    result = {
        "p_m_source": "CURRENT_SDF_CHILD",
        "q_m_source": "CURRENT_SDF_CHILD",
        "p_m_hash": _tensor_digest([("m", p_m)]),
        "q_m_hash": _tensor_digest([("m", q_m)]),
        "max_abs_m_difference": difference,
    }
    if difference >= 1e-7 or result["p_m_hash"] != result["q_m_hash"]:
        raise RuntimeError(f"P/Q current-M parity failed: {result}")
    return result


def function_snapshot(
    sdf_fc1: torch.nn.Module, parents: torch.Tensor, children: torch.Tensor
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        _, _, m, hatc, lnk = sdf_fc1.forward_step(
            x_prev=parents[:, 4:5],
            x_curr=children[:, :, 4:5],
            hatcf_prev=parents[:, 5:6],
            lnkf_prev=parents[:, 6:7],
            return_physical=True,
        )
    return {"m": m.detach(), "hatc": hatc.detach(), "lnk": lnk.detach()}


def drift_metrics(old: torch.Tensor, new: torch.Tensor, prefix: str) -> dict[str, float]:
    old_f = old.detach().reshape(-1).to(torch.float64)
    new_f = new.detach().reshape(-1).to(torch.float64)
    delta = new_f - old_f
    absolute = delta.abs()
    denominator = float(old_f.abs().mean()) + 1e-12
    centered_old = old_f - old_f.mean()
    centered_new = new_f - new_f.mean()
    corr_den = torch.linalg.vector_norm(centered_old) * torch.linalg.vector_norm(centered_new)
    corr = float(torch.dot(centered_old, centered_new) / corr_den) if corr_den > 0 else float("nan")
    return {
        f"{prefix}_mae": float(absolute.mean()),
        f"{prefix}_nmae": float(absolute.mean()) / denominator,
        f"{prefix}_p50": float(torch.quantile(absolute, 0.50)),
        f"{prefix}_p90": float(torch.quantile(absolute, 0.90)),
        f"{prefix}_p99": float(torch.quantile(absolute, 0.99)),
        f"{prefix}_max": float(absolute.max()),
        f"{prefix}_corr": corr,
    }


def forecast_metrics(actual: np.ndarray, forecast: np.ndarray, prefix: str) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    forecast = np.asarray(forecast, dtype=float).reshape(-1)
    mask = np.isfinite(actual) & np.isfinite(forecast)
    actual, forecast = actual[mask], forecast[mask]
    if actual.size < 2:
        return {f"{prefix}_{name}": float("nan") for name in (
            "mae", "rmse", "nrmse", "corr", "slope", "bias"
        )}
    error = forecast - actual
    scale = float(np.std(actual)) + 1e-12
    variance = float(np.var(actual))
    slope = float(np.cov(actual, forecast, ddof=0)[0, 1] / variance) if variance > 0 else float("nan")
    return {
        f"{prefix}_mae": float(np.mean(np.abs(error))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(error ** 2))),
        f"{prefix}_nrmse": float(np.sqrt(np.mean(error ** 2)) / scale),
        f"{prefix}_corr": float(np.corrcoef(actual, forecast)[0, 1]),
        f"{prefix}_slope": slope,
        f"{prefix}_bias": float(np.mean(error)),
    }


def macro_diagnostics(macro: pd.DataFrame) -> dict[str, float]:
    metrics = {}
    metrics.update(forecast_metrics(macro["Hatc"], macro["hatcf"], "fc1_hatc"))
    metrics.update(forecast_metrics(macro["LnK"], macro["lnkf"], "fc1_lnk"))
    metrics.update({
        "sdf_M_mean": float(macro["M"].mean()),
        "sdf_M_p10": float(macro["M"].quantile(0.10)),
        "sdf_M_p90": float(macro["M"].quantile(0.90)),
        "investment_rate": float(
            macro["I_oper"].sum() / max(float(macro["K"].abs().sum()), 1e-12)
        ),
        "default_rate": float("nan"),
        "mean_firms": float(macro["n_firms"].mean()),
    })
    return metrics


def simulation_policy_metrics(firm: pd.DataFrame) -> dict[str, float]:
    parent = firm[firm["branch"] == -1]
    if parent.empty:
        parent = firm
    return {
        "investment_rate_firm": float(parent["Bar_i_executed"].mean()),
        "default_rate": float((parent["P"] <= 0.0).mean()),
        "b_mean": float(parent["b"].mean()),
        "z_mean": float(parent["z"].mean()),
    }


def path_drift(old: pd.DataFrame, new: pd.DataFrame) -> dict[str, float]:
    keys = ["path", "t", "branch"]
    columns = ["Hatc", "LnK", "C", "K"]
    joined = old[keys + columns].merge(
        new[keys + columns], on=keys, suffixes=("_old", "_new"), how="inner"
    )
    result = {"macro_path_overlap_rows": int(len(joined))}
    for name in columns:
        a = joined[f"{name}_old"].to_numpy(dtype=float)
        b = joined[f"{name}_new"].to_numpy(dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        denom = np.linalg.norm(a[mask]) + 1e-12
        result[f"macro_{name.lower()}_path_drift"] = (
            float(np.linalg.norm(b[mask] - a[mask]) / denom) if mask.any() else float("nan")
        )
    return result


def macro_target_drift(old: pd.DataFrame, new: pd.DataFrame) -> dict[str, float]:
    drift = path_drift(old, new)
    return {
        "firm_to_hatc_target_disturbance": drift["macro_hatc_path_drift"],
        "firm_to_lnk_target_disturbance": drift["macro_lnk_path_drift"],
        "firm_to_C_target_disturbance": drift["macro_c_path_drift"],
        "firm_to_K_target_disturbance": drift["macro_k_path_drift"],
    }


def possible_two_cycle(
    current: Mapping[str, torch.Tensor], previous: Mapping[str, torch.Tensor],
    two_back: Mapping[str, torch.Tensor], *, near: float = 0.01, far: float = 0.02,
) -> bool:
    one, two = [], []
    for key in ("m", "hatc", "lnk"):
        one.append(drift_metrics(previous[key], current[key], "x")["x_nmae"])
        two.append(drift_metrics(two_back[key], current[key], "x")["x_nmae"])
    return bool(max(two) < near and max(one) > far)


def _load_state_dict_from_checkpoint(path: Path, key: str) -> Mapping[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if key in payload:
        return payload[key]
    if isinstance(payload.get("models"), Mapping) and key in payload["models"]:
        return payload["models"][key]
    if all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload
    raise ValueError(f"checkpoint {path} does not contain {key}")


def save_outer_checkpoint(
    root: Path, outer: int, sdf_fc1: torch.nn.Module, firm: torch.nn.Module,
    metadata: Mapping[str, Any],
) -> Path:
    target = root / f"outer_{outer:03d}"
    target.mkdir(parents=True, exist_ok=True)
    torch.save(sdf_fc1.state_dict(), target / "sdf_fc1.pt")
    torch.save({"policy_value": _clone_state(firm), "outer": outer}, target / "firm.pt")
    (target / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return target


def load_outer_checkpoint(
    target: Path, sdf_fc1: torch.nn.Module, firm: torch.nn.Module
) -> dict[str, Any]:
    sdf_fc1.load_state_dict(torch.load(target / "sdf_fc1.pt", map_location="cpu", weights_only=True))
    firm.load_state_dict(
        torch.load(target / "firm.pt", map_location="cpu", weights_only=False)["policy_value"]
    )
    return json.loads((target / "metadata.json").read_text(encoding="utf-8"))


def _simulation_kwargs(episode: Any, transition_seed: int, entry_seed: int) -> dict[str, Any]:
    names = (
        "entry_mode", "entry_spec_version", "entry_capital_ratio", "entry_size_ratio",
        "entry_cost_max", "entry_dummy_i", "entry_inference_chunk_size",
        "consumption_aggregation_mode", "consumption_floor", "node_accounting_mode",
        "entry_independent_resource_validation_enabled",
        "preserve_global_rng_around_entry",
    )
    kwargs = {name: getattr(episode.hyperparams, name) for name in names}
    market_defaults = {
        "market_clearing_mode": "disabled",
        "market_clearing_coarse_points": 10,
        "market_clearing_refinement_points": 20,
        "market_clearing_max_refinement_rounds": 4,
        "market_clearing_support_lower": -8.0,
        "market_clearing_support_upper": 2.0,
        "market_clearing_support_padding": 2.0,
        "market_clearing_residual_tolerance": 1.0e-5,
        "market_clearing_width_tolerance": 1.0e-5,
        "market_clearing_candidate_chunk_size": 262144,
    }
    kwargs.update({
        name: getattr(episode.hyperparams, name, default)
        for name, default in market_defaults.items()
    })
    kwargs.update({
        "entry_rng_seed": entry_seed,
        "entry_rng_mode": "stable_economic_identity",
        "common_transition_seed": transition_seed,
        "transition_rng_mode": "stable_firm_identity",
        "bp_action_source": "head",
    })
    return kwargs


def make_initial_state(
    *, models: Mapping[str, torch.nn.Module], episode: Any, paths: int, firms: int,
    horizon: int, branch_num: int, seed: int, device: torch.device,
) -> dict[str, torch.Tensor]:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    sim = SimulateTS(
        models=dict(models), config=episode.config, n_paths=paths, group_size=firms,
        horizon=horizon, branch_num=branch_num, device=device,
        **_simulation_kwargs(episode, seed + 1, seed + 2),
    )
    return clone_state(_initialize_batched_state(sim, firms), device="cpu")


def run_simulation(
    *, models: Mapping[str, torch.nn.Module], episode: Any,
    initial_state: Mapping[str, torch.Tensor], paths: int, firms: int,
    horizon: int, branch_num: int, transition_seed: int, entry_seed: int,
    device: torch.device,
):
    # Deliberately omit FC2 even if the source checkpoint contains it.
    allowed = {name: model for name, model in models.items() if name in {"policy_value", "sdf_fc1", "dist_b"}}
    sim = SimulateTS(
        models=allowed, config=episode.config, n_paths=paths, group_size=firms,
        horizon=horizon, branch_num=branch_num, device=device,
        initial_state=clone_state(initial_state, device="cpu"),
        **_simulation_kwargs(episode, transition_seed, entry_seed),
    )
    return sim.simulate_tensor()


def _to_frame(table) -> pd.DataFrame:
    frame = table.to_dataframe()
    for name in ("path", "t", "branch", "ID", "n_firms"):
        if name in frame:
            frame[name] = np.rint(frame[name]).astype(np.int64)
    return frame


def train_macro_block(
    episode: Any, models: Mapping[str, torch.nn.Module], macro_table: Any,
    *, fc1_epochs: int, sdf_epochs: int, recursive_epochs: int,
    batch_size: int, branch_num: int, fc1_max_rounds: int,
    fc1_rollout_diagnostic: bool,
) -> tuple[dict[str, Any], float]:
    trainable = set_stage_trainability(models, "macro")
    if trainable != ["sdf_fc1"]:
        raise RuntimeError(f"illegal macro trainable modules: {trainable}")
    hp = episode.hyperparams
    hp.fc1_only_epochs = int(fc1_epochs)
    hp.sdf_true_only_epochs = int(sdf_epochs)
    hp.sdf_recursive_only_epochs = int(recursive_epochs)
    hp.fc1_max_rounds = int(fc1_max_rounds)
    hp.fc1_rollout_diagnostic_enabled = bool(fc1_rollout_diagnostic)
    episode.optimizers = {
        "sdf_fc1": build_optimizers({"sdf_fc1": models["sdf_fc1"]}, hp)["sdf_fc1"]
    }
    episode.tensor_macro = macro_table
    episode.df_macro = None
    episode.episode_id = max(1, int(episode.episode_id))
    summary: dict[str, Any] = {}
    started = time.perf_counter()
    gate = episode._run_sdf_recon_from_macro(
        summary, n_epochs=max(fc1_epochs + sdf_epochs + recursive_epochs, 1),
        batch_size=batch_size, log_interval=max(batch_size, 1), n_branches=branch_num,
    )
    elapsed = time.perf_counter() - started
    summary["gate_result"] = gate
    return summary, elapsed


def flatten_numeric(payload: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    result = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(flatten_numeric(value, name))
        elif isinstance(value, (int, float, bool)):
            result[name] = value
    return result


def first_numeric_like(flat: Mapping[str, Any], needles: Sequence[str]) -> float:
    """Return the first finite numeric diagnostic matching all requested tokens."""
    for name, value in flat.items():
        lowered = name.lower()
        if all(token.lower() in lowered for token in needles):
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
    return float("nan")


def _plot_summary(summary: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    groups = {
        "firm_residuals.png": ["p_residual_post_inner", "q_residual_post_inner"],
        "macro_function_drift.png": ["sdf_M_function_drift_nmae", "fc1_hatc_function_drift_nmae", "fc1_lnk_function_drift_nmae"],
        "policy_flips.png": ["investment_flip", "default_flip"],
        "macro_to_firm_disturbance.png": ["macro_to_p_disturbance", "macro_to_q_disturbance"],
        "macro_path_drift.png": ["macro_hatc_path_drift", "macro_lnk_path_drift", "macro_c_path_drift", "macro_k_path_drift"],
    }
    for filename, columns in groups.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        for column in columns:
            if column in summary:
                ax.plot(summary["outer"], summary[column], marker="o", label=column)
        ax.set(xlabel="macro outer", yscale="symlog")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=170)
        plt.close(fig)
    existing = [path for path in groups if (figures / path).is_file()]
    if existing:
        fig, axes = plt.subplots(len(existing), 1, figsize=(10, 4 * len(existing)))
        axes = np.atleast_1d(axes)
        for ax, filename in zip(axes, existing):
            image = plt.imread(figures / filename)
            ax.imshow(image)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(figures / "summary_panel.png", dpi=150)
        plt.close(fig)


def _plot_macro_paths(paths: Sequence[pd.DataFrame], figures: Path) -> None:
    for name in ("Hatc", "LnK", "C", "K"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for outer, frame in enumerate(paths):
            parent = frame[frame["branch"] == -1]
            series = parent.groupby("t")[name].mean()
            ax.plot(series.index, series.values, label=f"outer {outer}", alpha=0.8)
        ax.set(xlabel="t", ylabel=name)
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"macro_path_{name.lower()}.png", dpi=170)
        plt.close(fig)


def _plot_forecast_scatter(macro: pd.DataFrame, outer: int, figures: Path) -> None:
    rng = np.random.default_rng(20261017 + outer)
    for actual, forecast, label in (("Hatc", "hatcf", "hatc"), ("LnK", "lnkf", "lnk")):
        use = macro[[actual, forecast]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(use) > 10000:
            use = use.iloc[rng.choice(len(use), 10000, replace=False)]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(use[actual], use[forecast], s=3, alpha=0.2)
        lo = float(min(use.min()))
        hi = float(max(use.max()))
        ax.plot([lo, hi], [lo, hi], color="black", lw=1)
        ax.set(xlabel=f"realized {actual}", ylabel=f"forecast {forecast}", title=f"outer {outer}")
        fig.tight_layout()
        fig.savefig(figures / f"forecast_{label}_outer_{outer:03d}.png", dpi=170)
        plt.close(fig)


def _write_report(
    path: Path, *, args: argparse.Namespace, metadata: Mapping[str, Any],
    summary: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        table = "No completed outer."
    else:
        # Keep the runner self-contained; pandas.to_markdown requires the
        # optional ``tabulate`` package, which is absent on some GPU workers.
        display_columns = [
            name for name in (
                "outer", "p_residual_pre_inner", "p_residual_post_inner",
                "q_residual_pre_inner", "q_residual_post_inner",
                "sdf_M_function_drift_nmae", "fc1_hatc_function_drift_nmae",
                "fc1_lnk_function_drift_nmae", "macro_hatc_path_drift",
                "macro_lnk_path_drift", "macro_to_p_disturbance",
                "macro_to_q_disturbance", "investment_flip", "default_flip",
                "outer_total_seconds",
            ) if name in summary.columns
        ]
        rows = []
        for values in summary[display_columns].itertuples(index=False, name=None):
            rows.append("| " + " | ".join(
                f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
                for value in values
            ) + " |")
        table = "\n".join([
            "| " + " | ".join(display_columns) + " |",
            "| " + " | ".join("---" for _ in display_columns) + " |",
            *rows,
        ])
    cls = metadata.get("classification", "E. inconclusive due implementation/operator issue")
    lines = [
        "# Macro outer / firm inner P-Q diagnostic",
        "",
        "## Actual dependency graph",
        "",
        "```text",
        "SDF + FC1 --current child M/Hatcf/LnKF--> P + Q + BP/policy",
        "P + Q + BP/policy --simulation with realized aggregates--> macro targets",
        "macro targets --production SDF/FC1 losses--> SDF + FC1",
        "```",
        "",
        "FC1_C predicts next Hatc; FC1_K predicts next LnK. Simulation feeds current",
        "realized aggregates into FC1 to form the next forecast. FC2 has no edge.",
        "",
        "## FC2 exclusion audit",
        "",
        f"- fc2_forward_calls: `{metadata.get('fc2_forward_calls')}`",
        f"- fc2_trainable_parameters: `{metadata.get('fc2_trainable_parameters')}`",
        f"- hash invariant: `{metadata.get('fc2_hash_invariant')}`",
        "",
        "## M parity",
        "",
        f"```json\n{json.dumps(metadata.get('m_parity', {}), indent=2)}\n```",
        "",
        "## Outer summary",
        "",
        table,
        "",
        "## Q1-Q15 answers",
        "",
        f"1. P/Q same current M: {metadata.get('m_parity_passed')}",
        f"2. FC2 excluded: {metadata.get('fc2_hash_invariant') and metadata.get('fc2_forward_calls') == 0}",
        "3-12. See the residual, function-drift, forecast, path, cross-block, and policy columns above.",
        f"13. possible macro two-cycle: {metadata.get('possible_macro_two_cycle')}",
        f"14. joint downward trend within five updates: {metadata.get('joint_downward_trend')}",
        f"15. classification: **{cls}**",
        "",
        "All convergence thresholds are diagnostic only, not an economic accuracy theorem.",
        "",
        "## Run configuration",
        "",
        f"```json\n{json.dumps(vars(args), indent=2, default=str)}\n```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-firm-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-sdf-fc1-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--macro-outer-max", type=int, default=5)
    parser.add_argument("--inner-p-target", type=float, default=0.02)
    parser.add_argument("--inner-q-target", type=float, default=0.02)
    parser.add_argument("--train-limit", type=int, default=8192)
    parser.add_argument("--validation-limit", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--simulation-paths", type=int, default=400)
    parser.add_argument("--validation-paths", type=int, default=400)
    parser.add_argument("--firms-per-path", type=int, default=200)
    parser.add_argument("--simulation-horizon", type=int, default=100)
    parser.add_argument("--branch-num", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--disable-fc2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--q-solver", choices=("root",), default="root")
    parser.add_argument("--p-policy-outer-max", type=int, default=2)
    parser.add_argument("--p-adam-steps", type=int, default=50)
    parser.add_argument("--p-lbfgs-iterations", type=int, default=100)
    parser.add_argument("--q-adam-steps", type=int, default=50)
    parser.add_argument("--q-lbfgs-iterations", type=int, default=100)
    parser.add_argument("--adam-lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lbfgs-chunk-iterations", type=int, default=20)
    parser.add_argument("--fc1-epochs", type=int, default=10)
    parser.add_argument("--sdf-epochs", type=int, default=5)
    parser.add_argument("--recursive-epochs", type=int, default=0)
    parser.add_argument("--fc1-max-rounds", type=int, default=2)
    parser.add_argument("--macro-batch-size", type=int, default=4096)
    parser.add_argument(
        "--fc1-rollout-diagnostic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.macro_outer_max <= 5:
        raise ValueError("macro-outer-max must be in [0, 5]")
    if not args.disable_fc2:
        raise ValueError("this experiment requires --disable-fc2")
    device = torch.device(args.device)
    output = args.output_dir.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    checkpoints = output / "checkpoints"
    figures = output / "figures"
    for path in (output, checkpoints, figures):
        path.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    source_run = args.source_run.expanduser().resolve()
    source_metadata = source_run
    if not (source_metadata / "metadata" / "hyperparams.json").is_file():
        source_metadata = source_run / "source"
    episode, hp, models, _, _, _, _, provenance, _ = _load_setup(
        source_run, device, None, source_metadata,
        args.source_checkpoint.expanduser().resolve(),
    )
    firm = models["policy_value"]
    firm.load_state_dict(_load_state_dict_from_checkpoint(args.initial_firm_checkpoint, "policy_value"))
    if args.initial_sdf_fc1_checkpoint is not None:
        models["sdf_fc1"].load_state_dict(
            _load_state_dict_from_checkpoint(args.initial_sdf_fc1_checkpoint, "sdf_fc1")
        )
    if args.resume_from is not None:
        load_outer_checkpoint(args.resume_from, models["sdf_fc1"], firm)

    fc2 = models.get("fc2")
    fc2_before = module_hash(fc2)
    fc2_counter = {"calls": 0}
    install_fc2_guard(fc2, fc2_counter)
    set_stage_trainability(models, "frozen")

    train_payload = torch.load(source_run / "fixed_point/collocation_train.pt", map_location="cpu", weights_only=False)
    val_payload = torch.load(source_run / "fixed_point/collocation_val.pt", map_location="cpu", weights_only=False)
    child_payload = torch.load(source_run / "fixed_point/child_shock_bank.pt", map_location="cpu", weights_only=False)
    n_train = min(args.train_limit, int(train_payload["parent_states"].shape[0]))
    n_val = min(args.validation_limit, int(val_payload["parent_states"].shape[0]))
    train_parents = train_payload["parent_states"][:n_train, :7].to(device)
    val_parents = val_payload["parent_states"][:n_val, :7].to(device)
    base_train_children = child_payload["train"]["children"][:n_train].to(device)
    base_val_children = child_payload["validation"]["children"][:n_val].to(device)
    m_min = float(getattr(hp, "pv_m_clamp_min", 0.7))
    m_max = float(getattr(hp, "pv_m_clamp_max", 1.3))

    sim_models = {name: model for name, model in models.items() if name != "fc2"}
    train_initial = make_initial_state(
        models=sim_models, episode=episode, paths=args.simulation_paths,
        firms=args.firms_per_path, horizon=args.simulation_horizon,
        branch_num=args.branch_num, seed=args.seed + 1000, device=device,
    )
    val_initial = make_initial_state(
        models=sim_models, episode=episode, paths=args.validation_paths,
        firms=args.firms_per_path, horizon=args.simulation_horizon,
        branch_num=args.branch_num, seed=args.seed + 2000, device=device,
    )
    crn = {
        "train_initial_cross_section_hash": state_hash(train_initial),
        "validation_initial_cross_section_hash": state_hash(val_initial),
        "train_shock_bank_hash": stable_json_hash([args.seed + 1011, args.branch_num, args.simulation_horizon]),
        "validation_shock_bank_hash": stable_json_hash([args.seed + 2011, args.branch_num, args.simulation_horizon]),
        "train_branch_bank_hash": stable_json_hash([args.seed + 1011, "stable_firm_identity"]),
        "validation_branch_bank_hash": stable_json_hash([args.seed + 2011, "stable_firm_identity"]),
        "train_simulation_seed": args.seed + 1011,
        "validation_simulation_seed": args.seed + 2011,
    }
    if crn["train_initial_cross_section_hash"] == crn["validation_initial_cross_section_hash"]:
        raise RuntimeError("train and validation initial-state banks must be distinct")

    rows: list[dict[str, Any]] = []
    macro_paths: list[pd.DataFrame] = []
    function_states: list[dict[str, torch.Tensor]] = []
    firm_views = []
    previous_macro: Optional[pd.DataFrame] = None
    pending: dict[str, Any] = {}
    stop_reason = "macro_outer_limit"
    m_parity: dict[str, Any] = {}

    for outer in range(args.macro_outer_max + 1):
        outer_started = time.perf_counter()
        train_children, train_m_diag = refresh_transition_bank(
            models["sdf_fc1"], train_parents, base_train_children,
            m_min=m_min, m_max=m_max,
        )
        val_children, val_m_diag = refresh_transition_bank(
            models["sdf_fc1"], val_parents, base_val_children,
            m_min=m_min, m_max=m_max,
        )
        m_parity = audit_m_parity(episode, val_children)
        set_stage_trainability(models, "firm")
        p_pre_problem = build_problem(
            episode, firm, val_parents, val_children, args.batch_size,
            snapshot_batch_size=min(512, args.batch_size), require_legacy_m1=False,
        )
        q_pre_problem, _ = build_q_survival_problem(
            episode=episode, initial_model=firm, frozen_p_model=deepcopy(firm),
            parents=val_parents, children=val_children, batch_size=args.batch_size,
            m_mode="current_sdf_child", include_all_regimes=True,
        )
        p_pre = evaluate_problem(firm, p_pre_problem)
        q_pre = evaluate_q_problem(firm, episode, q_pre_problem)
        old_view = build_coupled_view(firm, p_pre_problem)

        p_started = time.perf_counter()
        p_model, p_history, _, _ = solve_p_block(
            model=firm, episode=episode, train_parents=train_parents,
            train_children=train_children, validation_parents=val_parents,
            validation_children=val_children, batch_size=args.batch_size,
            policy_outer_max=args.p_policy_outer_max, adam_steps=args.p_adam_steps,
            adam_lr=args.adam_lr, eval_every=args.eval_every,
            lbfgs_max_iter=args.p_lbfgs_iterations,
            lbfgs_chunk_iterations=args.lbfgs_chunk_iterations,
            target_residual=args.inner_p_target, require_legacy_m1=False,
        )
        p_seconds = time.perf_counter() - p_started
        q_model, q_info, _, _ = solve_q_block(
            model=p_model, episode=episode, train_parents=train_parents,
            train_children=train_children, validation_parents=val_parents,
            validation_children=val_children, batch_size=args.batch_size,
            adam_steps=args.q_adam_steps, adam_lr=args.adam_lr,
            eval_every=args.eval_every, lbfgs_max_iter=args.q_lbfgs_iterations,
            lbfgs_chunk_iterations=args.lbfgs_chunk_iterations,
            target_residual=args.inner_q_target, m_mode="current_sdf_child",
        )
        firm = q_model
        models["policy_value"] = firm
        p_post_problem = build_problem(
            episode, firm, val_parents, val_children, args.batch_size,
            snapshot_batch_size=min(512, args.batch_size), require_legacy_m1=False,
        )
        q_post_problem, _ = build_q_survival_problem(
            episode=episode, initial_model=firm, frozen_p_model=deepcopy(firm),
            parents=val_parents, children=val_children, batch_size=args.batch_size,
            m_mode="current_sdf_child", include_all_regimes=True,
        )
        p_post = evaluate_problem(firm, p_post_problem)
        q_post = evaluate_q_problem(firm, episode, q_post_problem)
        new_view = build_coupled_view(firm, p_post_problem)
        policy_move = coupled_transition_metrics(old_view, new_view)
        firm_views.append(new_view)
        if outer in {0, 1, 3, 5}:
            plot_level_maps(new_view, outer, figures / "firm_surfaces")

        set_stage_trainability(models, "frozen")
        sim_started = time.perf_counter()
        train_out = run_simulation(
            models=models, episode=episode, initial_state=train_initial,
            paths=args.simulation_paths, firms=args.firms_per_path,
            horizon=args.simulation_horizon, branch_num=args.branch_num,
            transition_seed=args.seed + 1011, entry_seed=args.seed + 1012, device=device,
        )
        val_out = run_simulation(
            models=models, episode=episode, initial_state=val_initial,
            paths=args.validation_paths, firms=args.firms_per_path,
            horizon=args.simulation_horizon, branch_num=args.branch_num,
            transition_seed=args.seed + 2011, entry_seed=args.seed + 2012, device=device,
        )
        simulate_seconds = time.perf_counter() - sim_started
        train_macro = _to_frame(train_out.macro)
        val_macro = _to_frame(val_out.macro)
        val_firm = _to_frame(val_out.firm)
        diagnostics = macro_diagnostics(val_macro)
        diagnostics.update(simulation_policy_metrics(val_firm))
        macro_paths.append(val_macro)
        _plot_forecast_scatter(val_macro, outer, figures)
        sample = val_macro.sample(min(len(val_macro), 20000), random_state=args.seed + outer)
        outer_dir = output / f"outer_{outer:03d}"
        outer_dir.mkdir(parents=True, exist_ok=True)
        try:
            sample.to_parquet(outer_dir / "statewise_validation.parquet", index=False)
        except (ImportError, ModuleNotFoundError):
            sample.to_csv(outer_dir / "statewise_validation.csv", index=False)
        pd.DataFrame([diagnostics]).to_csv(outer_dir / "simulation_summary.csv", index=False)

        path_metrics = path_drift(previous_macro, val_macro) if previous_macro is not None else {
            "macro_hatc_path_drift": 0.0, "macro_lnk_path_drift": 0.0,
            "macro_c_path_drift": 0.0, "macro_k_path_drift": 0.0,
            "macro_path_overlap_rows": len(val_macro),
        }
        firm_target_metrics = macro_target_drift(previous_macro, val_macro) if previous_macro is not None else {
            "firm_to_hatc_target_disturbance": 0.0,
            "firm_to_lnk_target_disturbance": 0.0,
            "firm_to_C_target_disturbance": 0.0,
            "firm_to_K_target_disturbance": 0.0,
        }
        row = {
            "outer": outer,
            "p_residual_pre_inner": p_pre["normalized_mae"],
            "q_residual_pre_inner": q_pre["normalized_mae"],
            "p_residual_post_inner": p_post["normalized_mae"],
            "q_residual_post_inner": q_post["normalized_mae"],
            "macro_to_p_disturbance": max(0.0, p_pre["normalized_mae"] - float(pending.get("previous_p_post", p_pre["normalized_mae"]))),
            "macro_to_q_disturbance": max(0.0, q_pre["normalized_mae"] - float(pending.get("previous_q_post", q_pre["normalized_mae"]))),
            "p_inner_repair": p_pre["normalized_mae"] - p_post["normalized_mae"],
            "q_inner_repair": q_pre["normalized_mae"] - q_post["normalized_mae"],
            "p0_function_drift": policy_move["p0_function_change_p90"],
            "pi_function_drift": policy_move["pi_function_change_p90"],
            "q_function_drift": policy_move["q_movement_p90"],
            "investment_flip": policy_move["investment_selector_flip_share"],
            "default_flip": policy_move["default_survival_flip_share"],
            "bp0_p90_change": policy_move["bp0_p90"],
            "bpI_p90_change": policy_move["bpI_p90"],
            "inner_p_seconds": p_seconds,
            "inner_q_seconds": q_info["seconds"],
            "simulate_seconds": simulate_seconds,
            "sdf_fc1_train_seconds": float(pending.get("sdf_fc1_train_seconds", 0.0)),
            "outer_total_seconds": (
                time.perf_counter() - outer_started
                + float(pending.get("sdf_fc1_train_seconds", 0.0))
            ),
            "sdf_train_loss": float(pending.get("sdf_train_loss", float("nan"))),
            "sdf_val_loss": float(pending.get("sdf_val_loss", float("nan"))),
            "sdf_moment_residual": float(pending.get("sdf_moment_residual", float("nan"))),
            "macro_gradient_norm": float(pending.get("macro_gradient_norm", float("nan"))),
            "macro_learning_rate": float(pending.get("macro_learning_rate", float("nan"))),
            "sdf_M_function_drift_nmae": float(pending.get("sdf_M_function_drift_nmae", 0.0)),
            "fc1_hatc_function_drift_nmae": float(pending.get("fc1_hatc_function_drift_nmae", 0.0)),
            "fc1_lnk_function_drift_nmae": float(pending.get("fc1_lnk_function_drift_nmae", 0.0)),
            **diagnostics, **path_metrics, **firm_target_metrics,
        }
        rows.append(row)
        pd.DataFrame(p_history).to_csv(outer_dir / "inner_summary.csv", index=False)
        snapshot = function_snapshot(models["sdf_fc1"], val_parents, base_val_children)
        function_states.append({name: value.cpu() for name, value in snapshot.items()})
        save_outer_checkpoint(
            checkpoints, outer, models["sdf_fc1"], firm,
            {"outer": outer, "row": row, "m_parity": m_parity, "crn": crn},
        )
        pd.DataFrame(rows).to_csv(output / "macro_outer_summary.csv", index=False)
        print(json.dumps({"macro_outer_complete": outer, **row}, sort_keys=True), flush=True)
        previous_macro = val_macro

        if not all(math.isfinite(float(row[name])) for name in (
            "p_residual_post_inner", "q_residual_post_inner"
        )):
            stop_reason = "nonfinite_firm_residual"
            break
        if len(function_states) >= 3 and possible_two_cycle(
            function_states[-1], function_states[-2], function_states[-3]
        ):
            stop_reason = "possible_macro_two_cycle"
            break
        if outer == args.macro_outer_max:
            break

        macro_before = function_snapshot(models["sdf_fc1"], val_parents, base_val_children)
        macro_summary, macro_seconds = train_macro_block(
            episode, models, train_out.macro, fc1_epochs=args.fc1_epochs,
            sdf_epochs=args.sdf_epochs, recursive_epochs=args.recursive_epochs,
            batch_size=args.macro_batch_size, branch_num=args.branch_num,
            fc1_max_rounds=args.fc1_max_rounds,
            fc1_rollout_diagnostic=args.fc1_rollout_diagnostic,
        )
        set_stage_trainability(models, "frozen")
        macro_after = function_snapshot(models["sdf_fc1"], val_parents, base_val_children)
        flat_macro = flatten_numeric(macro_summary)
        pending = {
            "previous_p_post": p_post["normalized_mae"],
            "previous_q_post": q_post["normalized_mae"],
            "sdf_fc1_train_seconds": macro_seconds,
            "sdf_train_loss": first_numeric_like(flat_macro, ("sdf", "total")),
            "sdf_val_loss": first_numeric_like(flat_macro, ("final", "sdf", "loss")),
            "sdf_moment_residual": first_numeric_like(flat_macro, ("signed_aio", "mean")),
            "macro_gradient_norm": first_numeric_like(flat_macro, ("grad_norm",)),
            "macro_learning_rate": first_numeric_like(flat_macro, ("learning_rate",)),
            **drift_metrics(macro_before["m"], macro_after["m"], "sdf_M_function_drift"),
            **drift_metrics(macro_before["hatc"], macro_after["hatc"], "fc1_hatc_function_drift"),
            **drift_metrics(macro_before["lnk"], macro_after["lnk"], "fc1_lnk_function_drift"),
        }
        (outer_dir / "macro_train_history.json").write_text(
            json.dumps(macro_summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        pd.DataFrame([flat_macro]).to_csv(
            outer_dir / "macro_train_history.csv", index=False
        )
        if not bool(macro_summary.get("gate_result", {}).get("safe_to_continue", True)):
            stop_reason = "macro_training_gate_failure"
            break

    summary = pd.DataFrame(rows)
    summary.to_csv(output / "macro_outer_summary.csv", index=False)
    _plot_summary(summary, figures)
    _plot_macro_paths(macro_paths, figures)
    fc2_after = module_hash(fc2)
    joint_down = bool(
        len(summary) >= 3
        and summary["p_residual_post_inner"].iloc[-1] <= summary["p_residual_post_inner"].iloc[0]
        and summary["q_residual_post_inner"].iloc[-1] <= summary["q_residual_post_inner"].iloc[0]
        and summary["sdf_M_function_drift_nmae"].iloc[-1] <= summary["sdf_M_function_drift_nmae"].iloc[1]
    )
    classification = (
        "A. full outer convergence strongly promising" if joint_down
        else "B. firm inner stable but macro block not yet stable"
        if len(summary) and summary[["p_residual_post_inner", "q_residual_post_inner"]].iloc[-1].max() < 0.02
        else "D. clear oscillation / divergence" if stop_reason == "possible_macro_two_cycle"
        else "E. inconclusive due implementation/operator issue"
    )
    metadata = {
        "args": vars(args), "source_git_commit": provenance.get("source_git_commit"),
        "stop_reason": stop_reason, "completed_macro_updates": max(0, len(rows) - 1),
        "m_parity": m_parity, "m_parity_passed": m_parity.get("max_abs_m_difference", 1.0) < 1e-7,
        "fc2_forward_calls": fc2_counter["calls"],
        "fc2_trainable_parameters": 0,
        "fc2_hash_before": fc2_before, "fc2_hash_after": fc2_after,
        "fc2_hash_invariant": fc2_before == fc2_after,
        "possible_macro_two_cycle": stop_reason == "possible_macro_two_cycle",
        "joint_downward_trend": joint_down, "classification": classification,
        "crn": crn,
        "macro_inputs": "FC1(current realized aggregate, x transition); FC2 excluded",
        "m_semantics": "one current SDF-derived child M, clamped once to P range, shared by P and Q",
    }
    if fc2_counter["calls"] != 0 or fc2_before != fc2_after:
        raise RuntimeError(f"FC2 exclusion invariant failed: {metadata}")
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    _write_report(report_path, args=args, metadata=metadata, summary=summary)
    print(summary.to_string(index=False), flush=True)
    print(f"stop_reason={stop_reason}", flush=True)
    print(f"artifacts={output}", flush=True)
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
