"""Distribution-diverse short-horizon data for the joint Firm--Macro solver.

This module changes data support and state bookkeeping only.  Economic node
processing, entry/exit, resource accounting, FC1/SDF forward, and leverage
resolution remain owned by :class:`data.simulate_ts.SimulateTS`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from analysis.convergence_transition import MacroTransitionContext
from data.simulate_ts_parallel import _initialize_batched_state


DISTRIBUTION_FAMILIES = (
    "low_leverage",
    "high_leverage",
    "positive_bz_correlation",
    "negative_bz_correlation",
    "bimodal_leverage",
    "capital_high_z",
    "capital_low_z",
    "on_policy",
)


@dataclass(frozen=True)
class DiverseShortHorizonConfig:
    n_initial_distributions: int
    firms_per_distribution: int
    simulation_horizon: int = 5
    burn_in: int = 1
    branch_num: int = 2
    seed: int = 20261018
    mode: str = "main_short_horizon_training"
    distribution_family_weights: Optional[Mapping[str, float]] = None

    def validate(self) -> None:
        if self.n_initial_distributions < 1 or self.firms_per_distribution < 2:
            raise ValueError("at least one distribution and two firms are required")
        if self.simulation_horizon < 1:
            raise ValueError("simulation_horizon must be positive")
        if not 0 <= self.burn_in < self.simulation_horizon:
            raise ValueError("burn_in must lie in [0, simulation_horizon)")
        if self.branch_num != 2:
            raise ValueError("SDF AiO short-horizon data requires exactly two child branches")
        if self.mode not in {"initial_transition_diagnostic", "main_short_horizon_training"}:
            raise ValueError(f"unsupported short-horizon mode {self.mode!r}")


def _family_schedule(config: DiverseShortHorizonConfig) -> list[str]:
    config.validate()
    if config.distribution_family_weights is None:
        return [
            DISTRIBUTION_FAMILIES[i % len(DISTRIBUTION_FAMILIES)]
            for i in range(config.n_initial_distributions)
        ]
    keys = list(config.distribution_family_weights)
    if not keys or any(key not in DISTRIBUTION_FAMILIES for key in keys):
        raise ValueError("distribution_family_weights contains an unknown family")
    weights = np.asarray([config.distribution_family_weights[key] for key in keys], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("distribution family weights must be finite and nonnegative")
    rng = np.random.default_rng(config.seed + 7)
    return list(rng.choice(keys, config.n_initial_distributions, p=weights / weights.sum()))


def _rank_correlated_b(z: np.ndarray, *, positive: bool, rng: np.random.Generator) -> np.ndarray:
    latent = (z - z.mean()) / max(z.std(), 1e-8)
    latent = (1.0 if positive else -1.0) * 0.85 * latent + 0.35 * rng.normal(size=z.size)
    ranks = np.argsort(np.argsort(latent)).astype(float)
    return np.clip((ranks + rng.random(z.size)) / z.size, 0.0, 1.0)


def _draw_cross_section(
    family: str, n: int, rng: np.random.Generator, *, zbar: float, zstd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = rng.normal(zbar, zstd, size=n)
    if family == "low_leverage":
        b = rng.beta(2.0, 7.0, size=n)
    elif family == "high_leverage":
        b = rng.beta(7.0, 2.0, size=n)
    elif family == "positive_bz_correlation":
        b = _rank_correlated_b(z, positive=True, rng=rng)
    elif family == "negative_bz_correlation":
        b = _rank_correlated_b(z, positive=False, rng=rng)
    elif family == "bimodal_leverage":
        high = rng.random(n) < 0.5
        b = np.where(high, rng.beta(8.0, 2.0, n), rng.beta(2.0, 8.0, n))
    else:
        # on_policy preserves a broad joint support when no empirical source is
        # supplied.  Production callers may replace these rows with a captured
        # on-policy initial cross section before simulation.
        b = np.clip(0.5 + 0.18 * (z - zbar) / max(zstd, 1e-8) + 0.22 * rng.normal(size=n), 0, 1)
    if family == "capital_high_z":
        logits = 1.75 * (z - z.mean()) / max(z.std(), 1e-8)
    elif family == "capital_low_z":
        logits = -1.75 * (z - z.mean()) / max(z.std(), 1e-8)
    else:
        logits = 0.35 * (z - z.mean()) / max(z.std(), 1e-8) + rng.normal(0, 0.25, n)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return b, z, weights


def make_diverse_initial_state(
    simulator: Any,
    config: DiverseShortHorizonConfig,
    *,
    on_policy_state: Optional[Mapping[str, torch.Tensor]] = None,
) -> tuple[Dict[str, torch.Tensor], pd.DataFrame]:
    """Return a full production-compatible initial state and metadata."""
    config.validate()
    if simulator.n_paths != config.n_initial_distributions:
        raise ValueError("simulator n_paths does not match n_initial_distributions")
    if simulator.group_size != config.firms_per_distribution:
        raise ValueError("simulator group_size does not match firms_per_distribution")
    state = _initialize_batched_state(simulator, config.firms_per_distribution)
    device = simulator.device
    rng = np.random.default_rng(config.seed)
    families = _family_schedule(config)
    rho_z = float(simulator.config.RHO_Z)
    zstd = float(simulator.config.SIGMA_Z) / np.sqrt(max(1.0 - rho_z ** 2, 1e-12))
    rows = []
    for path, family in enumerate(families):
        if family == "on_policy" and on_policy_state is not None:
            for name in ("b", "z", "K"):
                value = on_policy_state[name][path].detach().cpu().numpy()
                state[name][path] = torch.as_tensor(value, device=device, dtype=state[name].dtype)
            b = state["b"][path].detach().cpu().numpy()
            z = state["z"][path].detach().cpu().numpy()
            K = state["K"][path].detach().cpu().numpy()
        else:
            b, z, weights = _draw_cross_section(
                family, config.firms_per_distribution, rng,
                zbar=float(simulator.config.ZBAR), zstd=zstd,
            )
            total_k = float(np.exp(rng.normal(4.0, 0.8)))
            K = total_k * weights
            state["b"][path] = torch.as_tensor(b, device=device, dtype=state["b"].dtype)
            state["z"][path] = torch.as_tensor(z, device=device, dtype=state["z"].dtype)
            state["K"][path] = torch.as_tensor(K, device=device, dtype=state["K"].dtype)
            state["lnkf"][path] = float(np.log(total_k))
        # eta and i deliberately remain draws from the model's production initializer.
        order = np.argsort(K)
        top = order[max(0, int(0.9 * len(order))):]
        rows.append({
            "distribution_family": family,
            "distribution_id": path,
            "path_id": path,
            "lineage_id": path,
            "seed": config.seed + path,
            "mean_b": float(np.mean(b)), "std_b": float(np.std(b)),
            "mean_z": float(np.mean(z)), "std_z": float(np.std(z)),
            "corr_b_z": float(np.corrcoef(b, z)[0, 1]) if np.std(b) and np.std(z) else 0.0,
            "K_total": float(np.sum(K)), "K_mean": float(np.mean(K)),
            "capital_concentration": float(np.sum(K[top]) / max(np.sum(K), 1e-12)),
            "capital_weighted_b": float(np.sum(K * b) / max(np.sum(K), 1e-12)),
            "capital_weighted_z": float(np.sum(K * z) / max(np.sum(K), 1e-12)),
            "b_p10": float(np.quantile(b, 0.10)), "b_p50": float(np.quantile(b, 0.50)),
            "b_p90": float(np.quantile(b, 0.90)), "z_p10": float(np.quantile(z, 0.10)),
            "z_p50": float(np.quantile(z, 0.50)), "z_p90": float(np.quantile(z, 0.90)),
            "leverage_high_tail_share": float(np.mean(b >= 0.8)),
            "capital_top10_share": float(np.sum(K[top]) / max(np.sum(K), 1e-12)),
            "firm_count": int(len(b)),
        })
    if (state["K"] < 0).any() or not torch.isfinite(state["K"]).all():
        raise RuntimeError("diverse initializer created invalid capital")
    return state, pd.DataFrame(rows)


PAIR_COLUMNS = (
    "path_id", "distribution_id", "lineage_id", "parent_t", "child_t", "branch_id",
    "x_t", "x_t1", "Hatcf_t", "LnKF_t", "Hatc_cal_t", "LnK_cal_t",
    "Hatcf_t1", "LnKF_t1", "Hatc_cal_t1", "LnK_cal_t1", "M_raw",
    "resource_feasible", "n_firms", "K_total", "C_total", "economic_node_id",
)


def build_short_horizon_pairs(
    macro_frame: pd.DataFrame,
    distribution_metadata: pd.DataFrame,
    config: DiverseShortHorizonConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build explicit parent/child economic-node pairs with discard reasons."""
    config.validate()
    required = {
        "path", "t", "branch", "x", "hatcf", "lnkf", "Hatc", "LnK", "M",
        "resource_feasible", "n_firms", "K", "C", "economic_node_id",
    }
    missing = sorted(required - set(macro_frame))
    if missing:
        raise KeyError(f"macro table missing pair columns: {missing}")
    parent = macro_frame[macro_frame["branch"] == -1].copy()
    child = macro_frame[macro_frame["branch"] >= 0].copy()
    child["parent_t"] = child["t"].astype(int) - 1
    parent = parent.rename(columns={
        "t": "parent_t", "x": "x_t", "hatcf": "Hatcf_t", "lnkf": "LnKF_t",
        "Hatc": "Hatc_cal_t", "LnK": "LnK_cal_t",
        "economic_node_id": "economic_node_id",
    })
    child = child.rename(columns={
        "t": "child_t", "branch": "branch_id", "x": "x_t1",
        "hatcf": "Hatcf_t1", "lnkf": "LnKF_t1", "Hatc": "Hatc_cal_t1",
        "LnK": "LnK_cal_t1", "M": "M_raw", "K": "K_total", "C": "C_total",
    })
    parent_cols = [
        "path", "parent_t", "x_t", "Hatcf_t", "LnKF_t", "Hatc_cal_t",
        "LnK_cal_t", "economic_node_id",
    ]
    child_cols = [
        "path", "parent_t", "child_t", "branch_id", "x_t1", "Hatcf_t1",
        "LnKF_t1", "Hatc_cal_t1", "LnK_cal_t1", "M_raw", "resource_feasible",
        "n_firms", "K_total", "C_total",
    ]
    pairs = child[child_cols].merge(parent[parent_cols], on=["path", "parent_t"], how="left", validate="many_to_one")
    pairs = pairs.merge(
        distribution_metadata[["path_id", "distribution_id", "lineage_id"]],
        left_on="path", right_on="path_id", how="left", validate="many_to_one",
    )
    pairs.drop(columns=["path"], inplace=True)
    pairs["child_t_valid"] = pairs["child_t"] == pairs["parent_t"] + 1
    pair_counts = pairs.groupby(["path_id", "parent_t"])["branch_id"].agg(
        lambda value: tuple(sorted(set(int(x) for x in value)))
    )
    valid_keys = set(pair_counts[pair_counts == tuple(range(config.branch_num))].index)
    pairs["branch_pair_valid"] = [
        (int(path), int(t)) in valid_keys for path, t in zip(pairs["path_id"], pairs["parent_t"])
    ]
    min_parent_t = 0 if config.mode == "initial_transition_diagnostic" else int(config.burn_in)
    pairs["burn_in_valid"] = pairs["parent_t"] >= min_parent_t
    pairs["finite_valid"] = np.isfinite(
        pairs[["x_t", "x_t1", "Hatcf_t", "LnKF_t", "Hatc_cal_t", "LnK_cal_t",
               "Hatcf_t1", "LnKF_t1", "Hatc_cal_t1", "LnK_cal_t1", "M_raw"]].to_numpy(float)
    ).all(axis=1)
    keep = pairs[["child_t_valid", "branch_pair_valid", "burn_in_valid", "finite_valid"]].all(axis=1)
    reasons = np.select(
        [~pairs["child_t_valid"], ~pairs["branch_pair_valid"], ~pairs["burn_in_valid"], ~pairs["finite_valid"]],
        ["child_time_mismatch", "missing_branch_pair", "burn_in", "nonfinite"], default="kept",
    )
    quality = pd.DataFrame({"discard_reason": reasons}).value_counts().rename("rows").reset_index()
    quality["retained"] = quality["discard_reason"] == "kept"
    kept = pairs.loc[keep, list(PAIR_COLUMNS)].sort_values(
        ["path_id", "parent_t", "branch_id"]
    ).reset_index(drop=True)
    if not bool((kept["child_t"] == kept["parent_t"] + 1).all()):
        raise RuntimeError("parent/child time alignment failed")
    return kept, quality


def transition_parity(
    sdf_fc1: torch.nn.Module,
    pairs: pd.DataFrame,
    *,
    m_min: float,
    m_max: float,
    device: torch.device,
    atol: float = 2e-6,
) -> Dict[str, Any]:
    """Compare pair records against one direct production MacroForward call."""
    tensor = lambda name: torch.as_tensor(pairs[name].to_numpy(), device=device, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        _, _, m_raw, hatcf, lnkf = sdf_fc1.forward_step(
            x_prev=tensor("x_t"), x_curr=tensor("x_t1"),
            hatcf_prev=tensor("Hatc_cal_t"), lnkf_prev=tensor("LnK_cal_t"),
            return_physical=True,
        )
    expected = {
        "Hatcf_t1": tensor("Hatcf_t1"), "LnKF_t1": tensor("LnKF_t1"),
        "M_raw": tensor("M_raw"),
    }
    actual = {"Hatcf_t1": hatcf, "LnKF_t1": lnkf, "M_raw": m_raw}
    errors = {name: float((actual[name] - expected[name]).abs().max()) for name in expected}
    m_used = m_raw.clamp(float(m_min), float(m_max))
    result = {
        "semantic_input": "MacroForward(x_t,x_t1,Hatc_cal_t,LnK_cal_t)",
        "n_rows": int(len(pairs)), "max_abs_errors": errors,
        "M_used_min": float(m_used.min()), "M_used_max": float(m_used.max()),
        "single_shared_clamp": True,
        "passed": bool(all(value <= atol for value in errors.values())),
        "atol": float(atol),
    }
    if not result["passed"]:
        raise RuntimeError(f"production transition parity failed: {result}")
    return result


def old_vs_new_transition(
    sdf_fc1: torch.nn.Module,
    pairs: pd.DataFrame,
    *, m_min: float, m_max: float, device: torch.device,
) -> pd.DataFrame:
    tensor = lambda name: torch.as_tensor(pairs[name].to_numpy(), device=device, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        _, _, old_m, old_h, old_k = sdf_fc1.forward_step(
            x_prev=tensor("x_t"), x_curr=tensor("x_t1"),
            hatcf_prev=tensor("Hatcf_t"), lnkf_prev=tensor("LnKF_t"), return_physical=True,
        )
        _, _, new_m, new_h, new_k = sdf_fc1.forward_step(
            x_prev=tensor("x_t"), x_curr=tensor("x_t1"),
            hatcf_prev=tensor("Hatc_cal_t"), lnkf_prev=tensor("LnK_cal_t"), return_physical=True,
        )
    result = pairs[["path_id", "distribution_id", "parent_t", "branch_id", "Hatcf_t", "LnKF_t", "Hatc_cal_t", "LnK_cal_t"]].copy()
    for name, value in {
        "old_child_Hatcf": old_h, "new_child_Hatcf": new_h,
        "old_child_LnKF": old_k, "new_child_LnKF": new_k,
        "old_M_raw": old_m, "new_M_raw": new_m,
        "old_M_used": old_m.clamp(m_min, m_max), "new_M_used": new_m.clamp(m_min, m_max),
    }.items():
        result[name] = value.detach().cpu().numpy().reshape(-1)
    return result


def save_provenance(path: Path, config: DiverseShortHorizonConfig, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {"config": asdict(config), **dict(payload)}
    path.write_text(json.dumps(content, indent=2, sort_keys=True, default=str), encoding="utf-8")


def dataframe_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()
