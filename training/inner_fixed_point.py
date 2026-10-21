"""Firm-side fixed-point iteration under a frozen macro/SDF operator.

This module deliberately reuses :class:`Episode` and :class:`BPGridTeacher`
for the economic operators.  It owns only immutable collocation/shock banks,
Gauss--Seidel sweep orchestration, invariant checks, and fixed-bank metrics.
It never simulates a long path and never trains the macro/SDF models.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from analysis.convergence_transition import (
    ConvergenceShockBank,
    MacroTransitionContext,
    build_child_exogenous_bundle,
)
from analysis.economic_config import AnalysisEconomicConfig
from losses.q_loss import classify_q_parent_regimes
from training.bp_grid_teacher import BPGridTeacher
from training.episode import Episode


STATE_COLUMNS = ("b", "z", "eta", "i", "x", "Hatcf", "LnKF")


def _tensor_hash(named_tensors: Iterable[Tuple[str, torch.Tensor]]) -> str:
    digest = sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def module_hash(module: nn.Module) -> str:
    return _tensor_hash(sorted(module.state_dict().items()))


def named_parameter_hash(model: nn.Module, prefixes: Sequence[str]) -> str:
    selected = [
        (name, value)
        for name, value in model.state_dict().items()
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    ]
    if not selected:
        raise ValueError(f"No model state matched prefixes {tuple(prefixes)}")
    return _tensor_hash(sorted(selected))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class InnerFPCollocationBank:
    parent_states: torch.Tensor
    source_kind: torch.Tensor
    seed: int
    split: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.parent_states.ndim != 2 or self.parent_states.shape[1] != 7:
            raise ValueError("parent_states must have shape [N,7]")
        if self.source_kind.shape != (self.parent_states.shape[0],):
            raise ValueError("source_kind must have shape [N]")
        if not torch.isfinite(self.parent_states).all():
            raise ValueError("collocation bank contains non-finite states")

    @property
    def fingerprint(self) -> str:
        return _tensor_hash((
            ("parent_states", self.parent_states),
            ("source_kind", self.source_kind),
        ))

    def payload(self) -> Dict[str, Any]:
        return {
            "parent_states": self.parent_states.detach().cpu(),
            "source_kind": self.source_kind.detach().cpu(),
            "seed": int(self.seed),
            "split": self.split,
            "metadata": dict(self.metadata),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class InnerFPChildBank:
    children: torch.Tensor
    eps_x: torch.Tensor
    eps_z: torch.Tensor
    u_eta: torch.Tensor
    u_i: torch.Tensor
    seed: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.children.ndim != 3 or self.children.shape[2] != 8:
            raise ValueError("children must have shape [N,J,8]")
        expected = self.children.shape[:2] + (1,)
        for name in ("eps_x", "eps_z", "u_eta", "u_i"):
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if not torch.isfinite(self.children).all():
            raise ValueError("child bank contains non-finite values")

    @property
    def fingerprint(self) -> str:
        return _tensor_hash((
            ("children", self.children),
            ("eps_x", self.eps_x),
            ("eps_z", self.eps_z),
            ("u_eta", self.u_eta),
            ("u_i", self.u_i),
        ))

    def payload(self) -> Dict[str, Any]:
        return {
            "children": self.children.detach().cpu(),
            "eps_x": self.eps_x.detach().cpu(),
            "eps_z": self.eps_z.detach().cpu(),
            "u_eta": self.u_eta.detach().cpu(),
            "u_i": self.u_i.detach().cpu(),
            "seed": int(self.seed),
            "metadata": dict(self.metadata),
            "fingerprint": self.fingerprint,
        }


def _finite_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    selected = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return selected.loc[np.isfinite(selected.to_numpy(dtype=np.float64)).all(axis=1)]


def _stratified_unit(n: int, rng: np.random.Generator) -> np.ndarray:
    values = (np.arange(n, dtype=np.float64) + rng.random(n)) / float(max(n, 1))
    return values[rng.permutation(n)]


def build_collocation_bank(
    firm_frame: pd.DataFrame,
    macro_frame: pd.DataFrame,
    *,
    n_parents: int,
    seed: int,
    split: str,
    economic_config: AnalysisEconomicConfig,
    on_support_share: float = 0.60,
    coverage_expansion: float = 0.15,
) -> InnerFPCollocationBank:
    """Build a deterministic joint-macro, broad-micro parent bank."""
    if n_parents < 2:
        raise ValueError("n_parents must be at least 2")
    if not 0.0 < on_support_share < 1.0:
        raise ValueError("on_support_share must be in (0,1)")
    rng = np.random.default_rng(int(seed))

    firm = firm_frame.rename(columns={"ETA": "eta"})
    macro = macro_frame.rename(columns={"hatcf": "Hatcf", "lnkf": "LnKF"})
    macro_support = _finite_frame(macro, ("x", "Hatcf", "LnKF"))
    z_support = _finite_frame(firm, ("z",))
    if macro_support.empty or z_support.empty:
        raise ValueError("Episode-1 artifacts do not contain finite macro/z support")

    macro_values = macro_support.to_numpy(dtype=np.float64)
    z_values = z_support["z"].to_numpy(dtype=np.float64)
    n_on = int(round(n_parents * on_support_share))
    n_cov = n_parents - n_on
    on_idx = rng.integers(0, macro_values.shape[0], size=n_on)
    on_macro = macro_values[on_idx]

    q01 = np.quantile(macro_values, 0.01, axis=0)
    q99 = np.quantile(macro_values, 0.99, axis=0)
    robust_width = np.maximum(q99 - q01, 1e-6)
    cov_base = macro_values[rng.integers(0, macro_values.shape[0], size=n_cov)]
    standardized = (macro_values - np.median(macro_values, axis=0)) / robust_width
    if standardized.shape[0] > 100_000:
        standardized = standardized[rng.choice(standardized.shape[0], 100_000, replace=False)]
    corr = np.cov(standardized, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = corr + np.eye(3) * 1e-8
    noise = rng.multivariate_normal(
        np.zeros(3), corr * float(coverage_expansion) ** 2, size=n_cov
    )
    cov_macro = cov_base + noise * robust_width
    lower = q01 - coverage_expansion * robust_width
    upper = q99 + coverage_expansion * robust_width
    cov_macro = np.clip(cov_macro, lower, upper)
    macro_draws = np.concatenate([on_macro, cov_macro], axis=0)
    source_kind = np.concatenate([
        np.zeros(n_on, dtype=np.int64),
        np.ones(n_cov, dtype=np.int64),
    ])

    b = _stratified_unit(n_parents, rng)
    # Q0 is an actual boundary equation, not a limit as b -> 0.  Reserve a
    # deterministic slice at the exact boundary while retaining broad support.
    n_zero = max(1, int(round(0.05 * n_parents)))
    b[:n_zero] = 0.0
    if n_parents > n_zero:
        b[n_zero] = 1.0
    i = float(economic_config.I_THRESHOLD) * _stratified_unit(n_parents, rng)
    eta = np.tile(np.array([0.0, 1.0]), int(math.ceil(n_parents / 2)))[:n_parents]
    eta = eta[rng.permutation(n_parents)]

    z = z_values[rng.integers(0, z_values.size, size=n_parents)]
    if n_cov:
        stationary_std = float(economic_config.SIGMA_Z) / math.sqrt(
            max(1.0 - float(economic_config.RHO_Z) ** 2, 1e-12)
        )
        broad = rng.normal(float(economic_config.ZBAR), stationary_std, size=n_cov)
        z_lo = min(float(np.quantile(z_values, 0.005)), float(economic_config.ZBAR - 4 * stationary_std))
        z_hi = max(float(np.quantile(z_values, 0.995)), float(economic_config.ZBAR + 4 * stationary_std))
        z[n_on:] = np.clip(broad, z_lo, z_hi)

    parent = np.column_stack([
        b,
        z,
        eta,
        i,
        macro_draws[:, 0],
        macro_draws[:, 1],
        macro_draws[:, 2],
    ])
    order = rng.permutation(n_parents)
    parent = parent[order]
    source_kind = source_kind[order]
    metadata = {
        "definition": "episode1_joint_macro_support_plus_robust_coverage",
        "state_columns": list(STATE_COLUMNS),
        "n_parents": int(n_parents),
        "on_support_share": float(on_support_share),
        "coverage_expansion": float(coverage_expansion),
        "macro_q01": dict(zip(("x", "Hatcf", "LnKF"), q01.tolist())),
        "macro_q99": dict(zip(("x", "Hatcf", "LnKF"), q99.tolist())),
        "z_source_p005": float(np.quantile(z_values, 0.005)),
        "z_source_p995": float(np.quantile(z_values, 0.995)),
        "eta0_count": int((parent[:, 2] <= 0.5).sum()),
        "eta1_count": int((parent[:, 2] > 0.5).sum()),
        "exact_b0_count": int((parent[:, 0] == 0.0).sum()),
    }
    return InnerFPCollocationBank(
        parent_states=torch.as_tensor(parent, dtype=torch.float32),
        source_kind=torch.as_tensor(source_kind, dtype=torch.long),
        seed=int(seed),
        split=str(split),
        metadata=metadata,
    )


def build_child_bank(
    sdf_fc1: nn.Module,
    parent_bank: InnerFPCollocationBank,
    *,
    n_child_shocks: int,
    seed: int,
    economic_config: AnalysisEconomicConfig,
    device: torch.device,
) -> InnerFPChildBank:
    """Create one immutable continuous-shock bank; eta is integrated later exactly."""
    if n_child_shocks < 2:
        raise ValueError("n_child_shocks must be at least 2")
    parent = parent_bank.parent_states.to(device)
    shock = ConvergenceShockBank.create(
        parent.shape[0], n_child_shocks, seed=seed, device=device, dtype=parent.dtype
    )
    sdf_hash_before = module_hash(sdf_fc1)
    sdf_fc1.eval()
    with torch.inference_mode():
        bundle = build_child_exogenous_bundle(
            sdf_fc1,
            parent,
            MacroTransitionContext(
                hatc_cal=parent[:, 5:6],
                lnk_cal=parent[:, 6:7],
            ),
            shock,
            economic_config=economic_config,
        )
    if module_hash(sdf_fc1) != sdf_hash_before:
        raise RuntimeError("Frozen sdf_fc1 changed while building the child bank")
    n, j = parent.shape[0], int(n_child_shocks)
    children = torch.stack([
        parent[:, 0].reshape(n, 1).expand(-1, j),
        bundle.z_next[..., 0],
        torch.zeros_like(bundle.eta_next[..., 0]),
        bundle.i_next[..., 0],
        bundle.x_next[..., 0],
        bundle.hatcf_next[..., 0],
        bundle.lnkf_next[..., 0],
        bundle.m_raw[..., 0],
    ], dim=-1)
    return InnerFPChildBank(
        children=children.detach().cpu(),
        eps_x=shock.eps_x.detach().cpu(),
        eps_z=shock.eps_z.detach().cpu(),
        u_eta=shock.u_eta.detach().cpu(),
        u_i=shock.u_i.detach().cpu(),
        seed=int(seed),
        metadata={
            "definition": "fixed_one_step_continuous_shocks_eta_integrated_exactly",
            "n_parents": int(n),
            "continuous_child_shocks": int(j),
            "effective_eta_expanded_children": int(2 * j),
            "parent_fingerprint": parent_bank.fingerprint,
            "sdf_fc1_hash": sdf_hash_before,
        },
    )


def make_batches(
    parent_bank: InnerFPCollocationBank,
    child_bank: InnerFPChildBank,
    *,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    if parent_bank.parent_states.shape[0] != child_bank.children.shape[0]:
        raise ValueError("parent and child bank sizes differ")
    batches: List[Dict[str, torch.Tensor]] = []
    n = int(parent_bank.parent_states.shape[0])
    for start in range(0, n, int(batch_size)):
        stop = min(start + int(batch_size), n)
        child_tensor = child_bank.children[start:stop].to(device)
        batches.append({
            "parent": parent_bank.parent_states[start:stop].to(device),
            "children": list(child_tensor.unbind(dim=1)),
            "source_id": parent_bank.source_kind[start:stop].to(device),
            "source_index": torch.arange(start, stop, device=device),
        })
    return batches


def _summary(values: torch.Tensor) -> Dict[str, float]:
    finite = values.detach().reshape(-1)
    finite = finite[torch.isfinite(finite)].to(torch.float32)
    if finite.numel() == 0:
        return {key: float("nan") for key in ("mean", "p50", "p90", "p99", "max")}
    return {
        "mean": float(finite.mean().item()),
        "p50": float(torch.quantile(finite, 0.50).item()),
        "p90": float(torch.quantile(finite, 0.90).item()),
        "p99": float(torch.quantile(finite, 0.99).item()),
        "max": float(finite.max().item()),
    }


def _function_outputs(model: nn.Module, states: torch.Tensor) -> Dict[str, torch.Tensor]:
    with torch.inference_mode():
        p0, pi = model._value_outputs(states)
        q = model._q_claim_output(states)
        equity = model.forward_equity(states)
    return {
        "p0": p0.detach(),
        "pi": pi.detach(),
        "q": q.detach(),
        "phat": equity["Phat"].detach(),
        "p": equity["P"].detach(),
    }


def _change_summary(current: torch.Tensor, previous: torch.Tensor, scale: torch.Tensor) -> Dict[str, float]:
    raw = (current - previous).abs()
    normalized = raw / scale.detach().clamp_min(1e-8)
    result = _summary(raw)
    result.update({f"normalized_{key}": value for key, value in _summary(normalized).items()})
    return result


@dataclass
class BoundaryGrid:
    states: torch.Tensor
    b_values: torch.Tensor
    z_values: torch.Tensor
    anchors: torch.Tensor
    anchor_quantiles: Tuple[float, ...]

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (len(self.anchor_quantiles), 2, self.b_values.numel(), self.z_values.numel())


def build_boundary_grid(
    macro_frame: pd.DataFrame,
    economic_config: AnalysisEconomicConfig,
    *,
    b_points: int = 51,
    z_points: int = 81,
    anchor_quantiles: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> BoundaryGrid:
    macro = macro_frame.rename(columns={"hatcf": "Hatcf", "lnkf": "LnKF"})
    support = _finite_frame(macro, ("x", "Hatcf", "LnKF"))
    if support.empty:
        raise ValueError("No finite macro anchors for the default-boundary grid")
    order = np.argsort(support["x"].to_numpy())
    sorted_support = support.iloc[order]
    anchors = []
    for quantile in anchor_quantiles:
        index = min(len(sorted_support) - 1, int(round(float(quantile) * (len(sorted_support) - 1))))
        anchors.append(sorted_support.iloc[index].to_numpy(dtype=np.float32))
    anchor_tensor = torch.as_tensor(np.stack(anchors), dtype=torch.float32)
    stationary_std = float(economic_config.SIGMA_Z) / math.sqrt(
        max(1.0 - float(economic_config.RHO_Z) ** 2, 1e-12)
    )
    b_values = torch.linspace(0.0, 1.0, int(b_points))
    z_values = torch.linspace(
        float(economic_config.ZBAR - 4 * stationary_std),
        float(economic_config.ZBAR + 4 * stationary_std),
        int(z_points),
    )
    states = []
    i_mid = 0.5 * float(economic_config.I_THRESHOLD)
    for anchor in anchor_tensor:
        for eta in (0.0, 1.0):
            bb, zz = torch.meshgrid(b_values, z_values, indexing="ij")
            n = bb.numel()
            states.append(torch.stack([
                bb.reshape(-1), zz.reshape(-1), torch.full((n,), eta),
                torch.full((n,), i_mid), torch.full((n,), float(anchor[0])),
                torch.full((n,), float(anchor[1])), torch.full((n,), float(anchor[2])),
            ], dim=1))
    return BoundaryGrid(
        states=torch.cat(states, dim=0),
        b_values=b_values,
        z_values=z_values,
        anchors=anchor_tensor,
        anchor_quantiles=tuple(float(q) for q in anchor_quantiles),
    )


def _default_cutoffs(phat: torch.Tensor, grid: BoundaryGrid) -> torch.Tensor:
    surface = phat.reshape(grid.shape)
    cutoffs = torch.full(grid.shape[:-1], float("nan"), dtype=torch.float32)
    z_values = grid.z_values
    for anchor in range(grid.shape[0]):
        for eta in range(grid.shape[1]):
            for b_index in range(grid.shape[2]):
                line = surface[anchor, eta, b_index]
                signs = line <= 0.0
                transitions = torch.nonzero(signs[:-1] != signs[1:], as_tuple=False).reshape(-1)
                if transitions.numel() == 0:
                    continue
                index = int(transitions[0].item())
                y0, y1 = float(line[index]), float(line[index + 1])
                z0, z1 = float(z_values[index]), float(z_values[index + 1])
                weight = 0.5 if y1 == y0 else min(max(-y0 / (y1 - y0), 0.0), 1.0)
                cutoffs[anchor, eta, b_index] = z0 + weight * (z1 - z0)
    return cutoffs


def evaluate_boundary(
    model: nn.Module,
    grid: BoundaryGrid,
    *,
    device: torch.device,
    batch_size: int = 8192,
) -> Dict[str, torch.Tensor]:
    values: Dict[str, List[torch.Tensor]] = {name: [] for name in ("phat", "p", "p0", "pi", "q")}
    model.eval()
    with torch.inference_mode():
        for states in grid.states.split(int(batch_size)):
            out = _function_outputs(model, states.to(device))
            for name in values:
                values[name].append(out[name].detach().cpu())
    result = {name: torch.cat(parts).reshape(grid.shape) for name, parts in values.items()}
    result["default"] = result["phat"] <= 0.0
    result["cutoff"] = _default_cutoffs(result["phat"], grid)
    return result


def boundary_metrics(
    current: Mapping[str, torch.Tensor],
    previous: Optional[Mapping[str, torch.Tensor]],
) -> Dict[str, float]:
    metrics = {"default_area_share": float(current["default"].float().mean().item())}
    if previous is None:
        metrics.update({
            "default_flip_share": float("nan"),
            "default_cutoff_drift_mean": float("nan"),
            "default_cutoff_drift_p90": float("nan"),
            "default_cutoff_drift_max": float("nan"),
            "default_cutoff_overlap_count": 0,
        })
        return metrics
    metrics["default_flip_share"] = float(
        (current["default"] != previous["default"]).float().mean().item()
    )
    valid = torch.isfinite(current["cutoff"]) & torch.isfinite(previous["cutoff"])
    drift = (current["cutoff"] - previous["cutoff"]).abs()[valid]
    summary = _summary(drift)
    metrics.update({
        "default_cutoff_drift_mean": summary["mean"],
        "default_cutoff_drift_p90": summary["p90"],
        "default_cutoff_drift_max": summary["max"],
        "default_cutoff_overlap_count": int(valid.sum().item()),
    })
    return metrics


@contextmanager
def _policy_value_model_scope(episode: Episode, model: nn.Module):
    """Temporarily make ``model`` the explicit current firm operator."""
    original = episode.models["policy_value"]
    episode.models["policy_value"] = model
    try:
        yield
    finally:
        episode.models["policy_value"] = original


def _operator_snapshot_metrics(snapshot: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    equations: Dict[str, Any] = {}
    for equation in ("p0", "pi", "q"):
        residual = (snapshot[f"{equation}_target"] - snapshot[f"{equation}_value"]).abs()
        normalized = residual / snapshot[f"{equation}_scale"].clamp_min(1e-8)
        equations[equation] = {
            "raw": _summary(residual),
            "normalized": _summary(normalized),
        }
    q_regimes: Dict[str, Any] = {}
    for index, name in enumerate(("zero", "default", "survival")):
        mask = snapshot["q_regime"] == index
        residual = (snapshot["q_target"] - snapshot["q_value"]).abs()[mask]
        normalized = residual / snapshot["q_scale"].clamp_min(1e-8)[mask]
        q_regimes[name] = {
            "count": int(mask.sum().item()),
            "raw": _summary(residual),
            "normalized": _summary(normalized),
        }
    mean_distance = max(equations[name]["normalized"]["mean"] for name in equations)
    p90_distance = max(equations[name]["normalized"]["p90"] for name in equations)
    return {
        "definition": "fresh_current_operator_snapshot_v1",
        "equations": equations,
        "q_regimes": q_regimes,
        "q_regime_counts": {name: q_regimes[name]["count"] for name in q_regimes},
        "q_recursion_gain": _summary(snapshot["q_recursion_gain"]),
        "q_recursion_gain_gt1_share": (
            float((snapshot["q_recursion_gain"] > 1.0).float().mean().item())
            if snapshot["q_recursion_gain"].numel() else float("nan")
        ),
        "empirical_fixed_point_distance_mean": float(mean_distance),
        "empirical_fixed_point_distance_p90": float(p90_distance),
    }


def evaluate_inner_operator_snapshot(
    episode: Episode,
    batches: List[Dict[str, torch.Tensor]],
    *,
    model: Optional[nn.Module] = None,
    include_bp_regret: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Fresh-evaluate outputs and Bellman targets on one fixed bank."""
    model = model or episode.models["policy_value"]
    model.eval()
    fields = (
        "p0_value", "pi_value", "q_value",
        "p0_target", "pi_target", "q_target",
        "p0_scale", "pi_scale", "q_scale", "q_regime",
    )
    pieces: Dict[str, List[torch.Tensor]] = {name: [] for name in fields}
    b0_star: List[torch.Tensor] = []
    bi_star: List[torch.Tensor] = []
    recursion_gain: List[torch.Tensor] = []
    teacher = BPGridTeacher.from_hyperparams(
        model, episode.loss_fns["p0"], episode.loss_fns["pi"],
        episode.hyperparams, q_target_model=model,
    )
    with _policy_value_model_scope(episode, model), torch.no_grad():
        for batch in batches:
            parent = batch["parent"][:, :7]
            children = episode._get_policy_children(batch)
            m_lo = float(getattr(episode.hyperparams, "pv_m_clamp_min", 0.7))
            m_hi = float(getattr(episode.hyperparams, "pv_m_clamp_max", 1.3))
            raw_m, used_m = episode._build_policy_m_lists(parent, children, m_lo, m_hi)
            children_exp, _, used_exp, weights = episode._expand_policy_expectation_children(
                children, raw_m, used_m
            )
            p0_result = teacher.compute_value_target(
                parent, children_exp, used_exp, branch="p0", child_weights=weights,
                **({"bp_pred": model.forward_policy(parent)[0]} if include_bp_regret else {})
            )
            pi_result = teacher.compute_value_target(
                parent, children_exp, used_exp, branch="pi", child_weights=weights,
                **({"bp_pred": model.forward_policy(parent)[1]} if include_bp_regret else {})
            )
            p0_value, pi_value = model._value_outputs(parent)
            value_scale = model.equity_value_scale(parent).detach().clamp_min(1e-8)

            equity = model.forward_equity(parent)
            masks = classify_q_parent_regimes(
                parent[:, 0:1], equity["Phat"],
                zero_b_eps=float(getattr(episode.hyperparams, "q_zero_b_eps", 0.0)),
            )
            q_value = model._q_claim_output(parent)
            q_target = torch.empty_like(q_value)
            q_scale = torch.empty_like(q_value)
            regime = torch.empty(parent.shape[0], dtype=torch.long, device=parent.device)

            zero = masks["zero"].reshape(-1)
            if zero.any():
                q_target[zero] = 0.0
                q_scale[zero] = 1.0
                regime[zero] = 0
            default = masks["default"].reshape(-1)
            if default.any():
                state = parent[default]
                target = episode.loss_fns["q"].compute_current_recovery(
                    state[:, 0:1], state[:, 4:5], state[:, 1:2]
                )
                q_target[default] = target
                q_scale[default] = 1.0 + target.abs()
                regime[default] = 1
            survival = masks["survival"].reshape(-1)
            if survival.any():
                subset = episode._slice_q_batch(batch, survival)
                state = subset["parent"][:, :7]
                child_list = episode._get_policy_children(subset)
                raw_q_m = [child[:, 7:8] for child in child_list]
                if getattr(episode.hyperparams, "q_use_detached_m", True):
                    q_lo = float(getattr(episode.hyperparams, "q_m_clamp_min", 0.5))
                    q_hi = float(getattr(episode.hyperparams, "q_m_clamp_max", 1.5))
                    used_q_m = [value.detach().clamp(q_lo, q_hi) for value in raw_q_m]
                else:
                    used_q_m = raw_q_m
                child_exp, _, used_q_m, _ = episode._expand_policy_expectation_children(
                    child_list, raw_q_m, used_q_m
                )
                output_t = model(state)
                bar_i = episode._policy_get_out(output_t, "bar_i", 5)
                multiplier = bar_i * (float(episode.loss_fns["q"].g) - 1.0) + 1.0
                b_sp = state[:, 0:1] / multiplier.clamp_min(1e-6)
                q_sp: List[torch.Tensor] = []
                bar_z_sp: List[torch.Tensor] = []
                for child in child_exp:
                    child_state = child[:, :7].clone()
                    child_state[:, 0:1] = b_sp
                    output_sp = model(child_state)
                    q_sp.append(episode._policy_get_out(output_sp, "Q", 0))
                    bar_z_sp.append(episode._policy_get_out(output_sp, "bar_z", 6))
                residuals = episode.loss_fns["q"].compute_main_residual(
                    model._q_claim_output(state), state[:, 0:1], bar_i,
                    used_q_m, q_sp, bar_z_sp,
                    [child[:, 4:5] for child in child_exp],
                    [child[:, 1:2] for child in child_exp],
                )
                residuals = episode._collapse_policy_eta_pairs(residuals)
                signed = episode._stack_signed_branch_residuals(
                    residuals, expected_batch=state.shape[0],
                    device=episode.device, label="q_snapshot",
                )
                branch_targets = model._q_claim_output(state) + signed
                q_target[survival] = branch_targets.mean(dim=1, keepdim=True)
                q_scale[survival] = 1.0 + branch_targets.abs().mean(dim=1, keepdim=True)
                regime[survival] = 2
                gains = [
                    m * multiplier * (1.0 - bar_z)
                    for m, bar_z in zip(used_q_m, bar_z_sp)
                ]
                gains = episode._collapse_policy_eta_pairs(gains)
                recursion_gain.append(torch.stack(gains, dim=1).reshape(-1).cpu())

            pieces["p0_value"].append(p0_value.detach().cpu())
            pieces["pi_value"].append(pi_value.detach().cpu())
            pieces["q_value"].append(q_value.detach().cpu())
            pieces["p0_target"].append(p0_result["value_star"].detach().cpu())
            pieces["pi_target"].append(pi_result["value_star"].detach().cpu())
            pieces["q_target"].append(q_target.detach().cpu())
            pieces["p0_scale"].append(value_scale.cpu())
            pieces["pi_scale"].append(value_scale.cpu())
            pieces["q_scale"].append(q_scale.detach().cpu())
            pieces["q_regime"].append(regime.detach().cpu())
            b0_star.append(p0_result["bp_star"].detach().cpu())
            bi_star.append(pi_result["bp_star"].detach().cpu())
            if include_bp_regret:
                pieces.setdefault('bp0_regret',[]).append(p0_result['regret'].detach().cpu())
                pieces.setdefault('bpi_regret',[]).append(pi_result['regret'].detach().cpu())
    snapshot = {name: torch.cat(values) for name, values in pieces.items()}
    snapshot["q_recursion_gain"] = (
        torch.cat(recursion_gain) if recursion_gain else torch.empty(0)
    )
    metrics = _operator_snapshot_metrics(snapshot)
    mean_tol = getattr(episode.hyperparams, "bellman_conditional_mean_thresh", None)
    p90_tol = getattr(episode.hyperparams, "bellman_conditional_p90_thresh", None)
    configured = mean_tol is not None and p90_tol is not None
    metrics.update({
        "bellman_thresholds": {"mean": mean_tol, "p90": p90_tol},
        "bellman_gate_configured": configured,
        "bellman_passed": (
            bool(all(
                metrics["equations"][name]["normalized"]["mean"] < float(mean_tol)
                and metrics["equations"][name]["normalized"]["p90"] < float(p90_tol)
                for name in ("p0", "pi", "q")
            )) if configured else None
        ),
    })
    policies = {"b0_star": torch.cat(b0_star), "bi_star": torch.cat(bi_star)}
    return metrics, policies, snapshot


def evaluate_inner_operator(
    episode: Episode,
    batches: List[Dict[str, torch.Tensor]],
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Evaluate T(F)-F with the same current model on both sides."""
    metrics, policies, _ = evaluate_inner_operator_snapshot(episode, batches)
    metrics["definition"] = "current_online_fixed_bank_operator_residual_v1"
    return metrics, policies


def _nan_change_summary() -> Dict[str, float]:
    return {key: float("nan") for key in (
        "mean", "p50", "p90", "p99", "max",
        "normalized_mean", "normalized_p50", "normalized_p90",
        "normalized_p99", "normalized_max",
    )}


def _target_difference(
    current: torch.Tensor,
    previous: torch.Tensor,
    scale: torch.Tensor,
    *,
    include_signed: bool = False,
) -> Dict[str, float]:
    delta = current - previous
    result = _summary(delta.abs())
    result.update({
        f"normalized_{key}": value
        for key, value in _summary(delta.abs() / scale.clamp_min(1e-8)).items()
    })
    if include_signed:
        result["signed_mean"] = float(delta.mean().item())
    return result


def _fit_error(
    values: torch.Tensor,
    frozen_target: torch.Tensor,
    scale: torch.Tensor,
) -> Dict[str, float]:
    return _target_difference(values, frozen_target, scale)


def _q_fit_by_regime(
    values: torch.Tensor,
    frozen_target: torch.Tensor,
    scale: torch.Tensor,
    regimes: torch.Tensor,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "overall": _fit_error(values, frozen_target, scale),
    }
    for index, name in enumerate(("zero", "default", "survival")):
        mask = regimes == index
        result[name] = (
            _fit_error(values[mask], frozen_target[mask], scale[mask])
            if mask.any() else _nan_change_summary()
        )
        result[name]["count"] = int(mask.sum().item())
    return result


def _q_training_diagnostics(summary: Mapping[str, Any]) -> Dict[str, float]:
    phases = {item.get("phase"): item for item in summary.get("phases", [])}
    metrics = phases.get("survival", {}).get("metrics", {})
    return {
        "q_main": float(metrics.get("q_main", float("nan"))),
        "q_aio_stable_mean": float(metrics.get("q_aio_stable_mean", float("nan"))),
        "q_aio_u_abs_mean": float(metrics.get("q_aio_u_abs_mean", float("nan"))),
        "q_raw_grad_norm": float(metrics.get("q_raw_grad_norm", float("nan"))),
        "q_clipped_grad_norm": float(metrics.get("q_clipped_grad_norm", float("nan"))),
    }


@dataclass
class InnerFPSolverConfig:
    mode: str = "coupled"
    max_sweeps: int = 30
    min_sweeps: int = 3
    consecutive_passes: int = 3
    batch_size: int = 256
    p_epochs: int = 100
    damping: float = 1.0
    change_mean_tol: float = 1e-3
    change_p90_tol: float = 5e-3
    default_flip_tol: float = 1e-3
    bstar_p90_tol: float = 1e-2
    seed: int = 20261014


class InnerFixedPointSolver:
    P_PREFIXES = ("value_encoder", "v0_head", "vi_head", "barz_model", "bari_model")
    Q_PREFIXES = ("q_encoder", "q_head")
    BP_PREFIXES = ("policy_encoder", "bp0_head", "bpi_head")

    def __init__(
        self,
        *,
        episode: Episode,
        train_bank: InnerFPCollocationBank,
        validation_bank: InnerFPCollocationBank,
        train_children: InnerFPChildBank,
        validation_children: InnerFPChildBank,
        boundary_grid: BoundaryGrid,
        output_dir: Path,
        config: InnerFPSolverConfig,
        source_provenance: Mapping[str, Any],
    ) -> None:
        self.episode = episode
        self.model = episode.models["policy_value"]
        self.sdf_fc1 = episode.models["sdf_fc1"]
        self.train_bank = train_bank
        self.validation_bank = validation_bank
        self.train_children = train_children
        self.validation_children = validation_children
        self.boundary_grid = boundary_grid
        self.output_dir = Path(output_dir)
        self.config = config
        self.source_provenance = dict(source_provenance)
        self.train_batches = make_batches(
            train_bank, train_children, batch_size=config.batch_size, device=episode.device
        )
        self.validation_batches = make_batches(
            validation_bank, validation_children,
            batch_size=config.batch_size, device=episode.device,
        )
        self.sdf_hash = module_hash(self.sdf_fc1)
        self.bank_hashes = {
            "train_parent": train_bank.fingerprint,
            "validation_parent": validation_bank.fingerprint,
            "train_children": train_children.fingerprint,
            "validation_children": validation_children.fingerprint,
        }
        self.history: List[Dict[str, Any]] = []
        self.previous_values: Optional[Dict[str, torch.Tensor]] = None
        self.previous_previous_values: Optional[Dict[str, torch.Tensor]] = None
        self.previous_boundary: Optional[Dict[str, torch.Tensor]] = None
        self.previous_previous_boundary: Optional[Dict[str, torch.Tensor]] = None
        self.previous_policies: Optional[Dict[str, torch.Tensor]] = None
        self.consecutive_passes = 0
        self.previous_targets: Optional[Dict[str, torch.Tensor]] = None
        self.initial_component_hashes = self._component_hashes()
        self.initial_boundary: Optional[Dict[str, torch.Tensor]] = None
        self.initial_q_regime_counts: Optional[Dict[str, int]] = None

    def _component_hashes(self) -> Dict[str, str]:
        return {
            "p": named_parameter_hash(self.model, self.P_PREFIXES),
            "q": named_parameter_hash(self.model, self.Q_PREFIXES),
            "bp": named_parameter_hash(self.model, self.BP_PREFIXES),
            "sdf_fc1": module_hash(self.sdf_fc1),
        }

    def _assert_frozen_sdf(self) -> None:
        if module_hash(self.sdf_fc1) != self.sdf_hash:
            raise RuntimeError("Frozen sdf_fc1 hash changed during inner fixed-point solve")
        if any(parameter.requires_grad for parameter in self.sdf_fc1.parameters()):
            raise RuntimeError("Frozen sdf_fc1 unexpectedly has trainable parameters")

    def _assert_banks_immutable(self) -> None:
        current = {
            "train_parent": self.train_bank.fingerprint,
            "validation_parent": self.validation_bank.fingerprint,
            "train_children": self.train_children.fingerprint,
            "validation_children": self.validation_children.fingerprint,
        }
        if current != self.bank_hashes:
            raise RuntimeError("Inner fixed-point parent/child bank mutated")

    def _values(self) -> Dict[str, torch.Tensor]:
        pieces: Dict[str, List[torch.Tensor]] = {
            name: [] for name in ("p0", "pi", "q", "phat", "p")
        }
        self.model.eval()
        with torch.inference_mode():
            for states in self.validation_bank.parent_states.split(self.config.batch_size):
                output = _function_outputs(self.model, states.to(self.episode.device))
                for name, value in output.items():
                    pieces[name].append(value.cpu())
        return {name: torch.cat(values) for name, values in pieces.items()}

    def _function_changes(self, values: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        if self.previous_values is None:
            return {name: {key: float("nan") for key in (
                "mean", "p50", "p90", "p99", "max", "normalized_mean",
                "normalized_p50", "normalized_p90", "normalized_p99", "normalized_max",
            )} for name in ("p0", "pi", "q")}
        states = self.validation_bank.parent_states
        xz_scale = self.model.equity_value_scale(states.to(self.episode.device)).detach().cpu()
        return {
            "p0": _change_summary(values["p0"], self.previous_values["p0"], xz_scale),
            "pi": _change_summary(values["pi"], self.previous_values["pi"], xz_scale),
            "q": _change_summary(
                values["q"], self.previous_values["q"],
                1.0 + self.previous_values["q"].abs(),
            ),
        }

    @staticmethod
    def _policy_changes(
        policies: Mapping[str, torch.Tensor],
        previous: Optional[Mapping[str, torch.Tensor]],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name, values in policies.items():
            if previous is None:
                stats = {key: float("nan") for key in ("mean", "p50", "p90", "p99", "max")}
            else:
                stats = _summary((values - previous[name]).abs())
            stats["boundary_low_share"] = float((values <= 1e-8).float().mean().item())
            stats["boundary_high_share"] = float((values >= 1.0 - 1e-8).float().mean().item())
            result[name] = stats
        return result

    def _save_checkpoint(self, sweep: int, name: str, extra: Mapping[str, Any]) -> Path:
        path = self.output_dir / "checkpoints" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        initial_hashes = getattr(self, "initial_component_hashes", None)
        if initial_hashes is None:
            initial_hashes = (
                self._component_hashes() if hasattr(self, "sdf_fc1") else {}
            )
        torch.save({
            "format": "inner_fixed_point_v1",
            "sweep": int(sweep),
            "policy_value": self.model.state_dict(),
            "sdf_fc1_hash": self.sdf_hash,
            "bank_hashes": self.bank_hashes,
            "source_provenance": self.source_provenance,
            "solver_config": asdict(self.config),
            "history": self.history,
            "previous_values": self.previous_values,
            "previous_previous_values": self.previous_previous_values,
            "previous_boundary": self.previous_boundary,
            "previous_previous_boundary": self.previous_previous_boundary,
            "previous_policies": self.previous_policies,
            "previous_targets": getattr(self, "previous_targets", None),
            "initial_component_hashes": initial_hashes,
            "initial_boundary": getattr(self, "initial_boundary", None),
            "initial_q_regime_counts": getattr(self, "initial_q_regime_counts", None),
            "consecutive_passes": int(self.consecutive_passes),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            **dict(extra),
        }, path)
        return path

    def load_resume(self, path: Path) -> int:
        payload = torch.load(path, map_location=self.episode.device, weights_only=False)
        if payload.get("format") != "inner_fixed_point_v1":
            raise ValueError("Not an inner-fixed-point checkpoint")
        if payload.get("bank_hashes") != self.bank_hashes:
            raise ValueError("Resume checkpoint bank fingerprints do not match")
        if payload.get("sdf_fc1_hash") != self.sdf_hash:
            raise ValueError("Resume checkpoint frozen sdf_fc1 hash does not match")
        self.model.load_state_dict(payload["policy_value"], strict=True)
        self.history = list(payload.get("history", []))
        # Diagnostic evaluators deliberately return CPU tensors.  A CUDA
        # map_location is needed for model weights but must not migrate the
        # saved comparison surfaces, otherwise the first resumed sweep mixes
        # CPU current values with CUDA previous values.
        def _diagnostic_cpu(value):
            if value is None:
                return None
            return {
                key: tensor.detach().cpu() if torch.is_tensor(tensor) else tensor
                for key, tensor in value.items()
            }

        self.previous_values = _diagnostic_cpu(payload.get("previous_values"))
        self.previous_previous_values = _diagnostic_cpu(
            payload.get("previous_previous_values")
        )
        self.previous_boundary = _diagnostic_cpu(payload.get("previous_boundary"))
        self.previous_previous_boundary = _diagnostic_cpu(
            payload.get("previous_previous_boundary")
        )
        self.previous_policies = _diagnostic_cpu(payload.get("previous_policies"))
        self.previous_targets = _diagnostic_cpu(payload.get("previous_targets"))
        fallback_initial_hashes = getattr(self, "initial_component_hashes", None)
        if fallback_initial_hashes is None:
            fallback_initial_hashes = (
                self._component_hashes() if hasattr(self, "sdf_fc1") else {}
            )
        self.initial_component_hashes = dict(
            payload.get("initial_component_hashes", fallback_initial_hashes)
        )
        self.initial_boundary = _diagnostic_cpu(payload.get("initial_boundary"))
        initial_counts = payload.get("initial_q_regime_counts")
        self.initial_q_regime_counts = (
            {str(key): int(value) for key, value in initial_counts.items()}
            if initial_counts is not None else None
        )
        self.consecutive_passes = int(payload.get("consecutive_passes", 0))
        return int(payload["sweep"]) + 1

    def _save_static_artifacts(self) -> None:
        fp = self.output_dir / "fixed_point"
        fp.mkdir(parents=True, exist_ok=True)
        torch.save(self.train_bank.payload(), fp / "collocation_train.pt")
        torch.save(self.validation_bank.payload(), fp / "collocation_val.pt")
        torch.save({
            "train": self.train_children.payload(),
            "validation": self.validation_children.payload(),
        }, fp / "child_shock_bank.pt")
        torch.save({
            "anchors": self.boundary_grid.anchors,
            "anchor_quantiles": self.boundary_grid.anchor_quantiles,
            "b_values": self.boundary_grid.b_values,
            "z_values": self.boundary_grid.z_values,
            "states": self.boundary_grid.states,
        }, fp / "macro_anchor_bank.pt")
        write_json(fp / "bank_metadata.json", {
            "bank_hashes": self.bank_hashes,
            "train": self.train_bank.metadata,
            "validation": self.validation_bank.metadata,
            "train_children": self.train_children.metadata,
            "validation_children": self.validation_children.metadata,
        })

    def _write_metrics(self) -> None:
        fp = self.output_dir / "fixed_point"
        write_json(fp / "metrics.json", {"sweeps": self.history})
        flat_rows = []
        for row in self.history:
            flat: Dict[str, Any] = {}
            def visit(prefix: str, value: Any) -> None:
                if isinstance(value, Mapping):
                    for key, item in value.items():
                        visit(f"{prefix}_{key}" if prefix else str(key), item)
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    flat[prefix] = value
            visit("", row)
            flat_rows.append(flat)
        pd.DataFrame(flat_rows).to_csv(fp / "metrics.csv", index=False)
        block_rows: List[Dict[str, Any]] = []
        for row in self.history:
            equations = row["operator"]["equations"]
            q_regimes = row["operator"]["q_regimes"]
            changes = row["function_change"]
            fit = row["frozen_target_fit"]
            cross = row["cross_effects"]
            stages = row["operator_stages"]
            q_diag = row["q_training_diagnostics"]
            recursion = row["operator"]["q_recursion_gain"]
            block_rows.append({
                "mode": row["mode"],
                "sweep": row["sweep"],
                "p0_residual_mean": equations["p0"]["normalized"]["mean"],
                "p0_residual_p90": equations["p0"]["normalized"]["p90"],
                "p0_residual_p99": equations["p0"]["normalized"]["p99"],
                "p0_residual_raw_mean": equations["p0"]["raw"]["mean"],
                "p0_residual_raw_p90": equations["p0"]["raw"]["p90"],
                "pi_residual_mean": equations["pi"]["normalized"]["mean"],
                "pi_residual_p90": equations["pi"]["normalized"]["p90"],
                "pi_residual_p99": equations["pi"]["normalized"]["p99"],
                "pi_residual_raw_mean": equations["pi"]["raw"]["mean"],
                "pi_residual_raw_p90": equations["pi"]["raw"]["p90"],
                "q_residual_mean": equations["q"]["normalized"]["mean"],
                "q_residual_p90": equations["q"]["normalized"]["p90"],
                "q_residual_p99": equations["q"]["normalized"]["p99"],
                "q_residual_raw_mean": equations["q"]["raw"]["mean"],
                "q_residual_raw_p90": equations["q"]["raw"]["p90"],
                "q0_residual_mean": q_regimes["zero"]["normalized"]["mean"],
                "qd_residual_mean": q_regimes["default"]["normalized"]["mean"],
                "qs_residual_mean": q_regimes["survival"]["normalized"]["mean"],
                "qs_residual_p90": q_regimes["survival"]["normalized"]["p90"],
                "p0_function_change": changes["p0"]["normalized_p90"],
                "pi_function_change": changes["pi"]["normalized_p90"],
                "q_function_change": changes["q"]["normalized_p90"],
                "p0_function_change_mean": changes["p0"]["normalized_mean"],
                "pi_function_change_mean": changes["pi"]["normalized_mean"],
                "q_function_change_mean": changes["q"]["normalized_mean"],
                "p_fit_mean": max(
                    fit["p0"]["normalized_mean"], fit["pi"]["normalized_mean"]
                ),
                "p_fit_p90": max(
                    fit["p0"]["normalized_p90"], fit["pi"]["normalized_p90"]
                ),
                "p0_fit_mean": fit["p0"]["normalized_mean"],
                "p0_fit_p90": fit["p0"]["normalized_p90"],
                "pi_fit_mean": fit["pi"]["normalized_mean"],
                "pi_fit_p90": fit["pi"]["normalized_p90"],
                "q_fit_mean": fit["q"]["overall"]["normalized_mean"],
                "q_fit_p90": fit["q"]["overall"]["normalized_p90"],
                "q0_fit_mean": fit["q"]["zero"]["normalized_mean"],
                "qd_fit_mean": fit["q"]["default"]["normalized_mean"],
                "qs_fit_mean": fit["q"]["survival"]["normalized_mean"],
                "default_area": row["default_boundary"]["default_area_share"],
                "default_flip": row["default_boundary"]["default_flip_share"],
                "q_zero_count": row["operator"]["q_regime_counts"]["zero"],
                "q_default_count": row["operator"]["q_regime_counts"]["default"],
                "q_survival_count": row["operator"]["q_regime_counts"]["survival"],
                **q_diag,
                "q_recursion_gain_mean": recursion["mean"],
                "q_recursion_gain_p90": recursion["p90"],
                "q_recursion_gain_gt1_share": row["operator"]["q_recursion_gain_gt1_share"],
                "p_to_q_effect_mean": cross["p_to_q"]["mean"],
                "p_to_q_effect_p90": cross["p_to_q"]["p90"],
                "p_to_q_effect_normalized_mean": cross["p_to_q"]["normalized_mean"],
                "p_to_q_effect_normalized_p90": cross["p_to_q"]["normalized_p90"],
                "q_to_p_p0_effect_mean": cross["q_to_p0"]["mean"],
                "q_to_p_p0_effect_p90": cross["q_to_p0"]["p90"],
                "q_to_p_p0_effect_signed_mean": cross["q_to_p0"].get("signed_mean"),
                "q_to_p_p0_effect_normalized_mean": cross["q_to_p0"]["normalized_mean"],
                "q_to_p_p0_effect_normalized_p90": cross["q_to_p0"]["normalized_p90"],
                "q_to_p_pi_effect_mean": cross["q_to_pi"]["mean"],
                "q_to_p_pi_effect_p90": cross["q_to_pi"]["p90"],
                "q_to_p_pi_effect_signed_mean": cross["q_to_pi"].get("signed_mean"),
                "q_to_p_pi_effect_normalized_mean": cross["q_to_pi"]["normalized_mean"],
                "q_to_p_pi_effect_normalized_p90": cross["q_to_pi"]["normalized_p90"],
                "start_fp_residual": stages["start"]["empirical_fixed_point_distance_p90"],
                "post_p_fp_residual": stages["post_p"]["empirical_fixed_point_distance_p90"],
                "post_q_fp_residual": stages["post_q"]["empirical_fixed_point_distance_p90"],
                "start_p0_residual_p90": stages["start"]["equations"]["p0"]["normalized"]["p90"],
                "post_p_p0_residual_p90": stages["post_p"]["equations"]["p0"]["normalized"]["p90"],
                "post_q_p0_residual_p90": stages["post_q"]["equations"]["p0"]["normalized"]["p90"],
                "start_pi_residual_p90": stages["start"]["equations"]["pi"]["normalized"]["p90"],
                "post_p_pi_residual_p90": stages["post_p"]["equations"]["pi"]["normalized"]["p90"],
                "post_q_pi_residual_p90": stages["post_q"]["equations"]["pi"]["normalized"]["p90"],
                "start_q_residual_p90": stages["start"]["equations"]["q"]["normalized"]["p90"],
                "post_p_q_residual_p90": stages["post_p"]["equations"]["q"]["normalized"]["p90"],
                "post_q_q_residual_p90": stages["post_q"]["equations"]["q"]["normalized"]["p90"],
                "target_movement_p0_mean": row["target_movement"]["p0"]["normalized_mean"],
                "target_movement_p0_p90": row["target_movement"]["p0"]["normalized_p90"],
                "target_movement_pi_mean": row["target_movement"]["pi"]["normalized_mean"],
                "target_movement_pi_p90": row["target_movement"]["pi"]["normalized_p90"],
                "target_movement_q_mean": row["target_movement"]["q"]["normalized_mean"],
                "target_movement_q_p90": row["target_movement"]["q"]["normalized_p90"],
            })
        pd.DataFrame(block_rows).to_csv(fp / "block_isolation_metrics.csv", index=False)

    def _failure_reasons(self, latest: Mapping[str, Any]) -> List[str]:
        reasons = []
        if latest["operator"]["bellman_passed"] is None:
            reasons.append("convergence_gate_unconfigured")
        elif latest["operator"]["bellman_passed"] is False:
            reasons.append("bellman_residual_not_converged")
        if not latest["gates"]["function_change_passed"]:
            reasons.append("function_change_not_converged")
        if not latest["gates"]["default_boundary_passed"]:
            reasons.append("default_boundary_not_stable")
        if not latest["gates"]["implicit_policy_passed"]:
            reasons.append("implicit_policy_not_stable")
        if latest["gates"].get("two_cycle_detected"):
            reasons.append("two_cycle_detected")
        return reasons

    def run(self, *, start_sweep: int = 1) -> Dict[str, Any]:
        if not 0.0 < float(self.config.damping) <= 1.0:
            raise ValueError("fp_damping must be in (0,1]")
        mode = str(self.config.mode).lower()
        if mode not in {"coupled", "p_only", "q_only"}:
            raise ValueError("mode must be one of: coupled, p_only, q_only")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_static_artifacts()
        self.sdf_fc1.eval().requires_grad_(False)
        if "fc2" in self.episode.models and self.episode.models["fc2"] is not None:
            self.episode.models["fc2"].eval().requires_grad_(False)
        self.episode._inner_fp_target_damping = float(self.config.damping)
        self._assert_frozen_sdf()
        self._assert_banks_immutable()

        if self.previous_values is None:
            self.previous_values = self._values()
            self.previous_boundary = evaluate_boundary(
                self.model, self.boundary_grid, device=self.episode.device
            )
            initial_operator, self.previous_policies, initial_snapshot = (
                evaluate_inner_operator_snapshot(
                self.episode, self.validation_batches
                )
            )
            self.initial_boundary = {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in self.previous_boundary.items()
            }
            self.initial_q_regime_counts = dict(initial_operator["q_regime_counts"])
            boundary_dir = self.output_dir / "fixed_point" / "boundaries"
            surface_dir = self.output_dir / "fixed_point" / "surfaces"
            boundary_dir.mkdir(parents=True, exist_ok=True)
            surface_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.previous_boundary, boundary_dir / "sweep_000.pt")
            torch.save(
                {
                    "values": self.previous_values,
                    "policies": self.previous_policies,
                    "operator_snapshot": initial_snapshot,
                },
                surface_dir / "sweep_000.pt",
            )
            self._save_checkpoint(0, "sweep_000.pt", {"stage": "initial"})

        passed = False
        first_pass_sweep: Optional[int] = None
        numerical_failure: Optional[str] = None
        for sweep in range(int(start_sweep), int(self.config.max_sweeps) + 1):
            sweep_start = time.perf_counter()
            random.seed(self.config.seed + sweep)
            np.random.seed((self.config.seed + sweep) % (2**32 - 1))
            torch.manual_seed(self.config.seed + sweep)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed + sweep)
            self._assert_frozen_sdf()
            self._assert_banks_immutable()
            hashes_start = self._component_hashes()
            teacher_k = deepcopy(self.model).to(self.episode.device).eval().requires_grad_(False)

            try:
                start_operator, _, start_snapshot = evaluate_inner_operator_snapshot(
                    self.episode, self.validation_batches, model=teacher_k
                )
                start_boundary = evaluate_boundary(
                    teacher_k, self.boundary_grid, device=self.episode.device
                )

                p_start = time.perf_counter()
                if mode == "q_only":
                    p_summary = {
                        "status": "frozen_block",
                        "optimizer_steps": 0,
                        "mode": mode,
                    }
                else:
                    p_summary = self.episode._run_policy_value_evaluation_stage(
                        self.train_batches,
                        self.validation_batches,
                        teacher_k,
                        int(self.config.p_epochs),
                        q_target_model=teacher_k,
                    )
                    if p_summary.get("status") != "accepted":
                        raise RuntimeError(
                            "P stage was not accepted: "
                            f"{p_summary.get('status', 'missing_status')}"
                        )
                p_seconds = time.perf_counter() - p_start
                hashes_post_p = self._component_hashes()
                if hashes_post_p["q"] != hashes_start["q"]:
                    raise RuntimeError("P stage changed Q parameters")
                if hashes_post_p["bp"] != hashes_start["bp"]:
                    raise RuntimeError("P stage changed BP parameters")
                if hashes_post_p["sdf_fc1"] != hashes_start["sdf_fc1"]:
                    raise RuntimeError("P stage changed frozen sdf_fc1")
                self._save_checkpoint(
                    sweep, f"sweep_{sweep:03d}_post_p.pt", {"stage": "post_p"}
                )

                p_snapshot = deepcopy(self.model).to(self.episode.device).eval().requires_grad_(False)
                post_p_operator, _, post_p_snapshot = evaluate_inner_operator_snapshot(
                    self.episode, self.validation_batches, model=p_snapshot
                )
                post_p_boundary = evaluate_boundary(
                    p_snapshot, self.boundary_grid, device=self.episode.device
                )
                q_start = time.perf_counter()
                if mode == "p_only":
                    q_summary = {
                        "status": "frozen_block",
                        "optimizer_steps": 0,
                        "phases": [],
                        "mode": mode,
                    }
                else:
                    q_summary = self.episode._run_q_regime_training(
                        self.train_batches,
                        p_snapshot,
                        validation_batches=self.validation_batches,
                        fixed_q_target_model=teacher_k,
                    )
                    if not str(q_summary.get("status", "")).startswith("accepted"):
                        raise RuntimeError(
                            "Q stage was not accepted: "
                            f"{q_summary.get('status', 'missing_status')} "
                            f"({q_summary.get('q_stage_rejection_reason')})"
                        )
                q_seconds = time.perf_counter() - q_start
                hashes_post_q = self._component_hashes()
                if hashes_post_q["p"] != hashes_post_p["p"]:
                    raise RuntimeError("Q stage changed P parameters")
                if hashes_post_q["bp"] != hashes_post_p["bp"]:
                    raise RuntimeError("Q stage changed BP parameters")
                if hashes_post_q["sdf_fc1"] != hashes_post_p["sdf_fc1"]:
                    raise RuntimeError("Q stage changed frozen sdf_fc1")

                eval_start = time.perf_counter()
                operator, policies, post_q_snapshot = evaluate_inner_operator_snapshot(
                    self.episode, self.validation_batches
                )
                values = self._values()
                changes = self._function_changes(values)
                policy_changes = self._policy_changes(policies, self.previous_policies)
                boundary = evaluate_boundary(
                    self.model, self.boundary_grid, device=self.episode.device
                )
                boundary_summary = boundary_metrics(boundary, self.previous_boundary)
                post_q_boundary = boundary
                fit = {
                    "p0": (
                        _fit_error(
                            post_p_snapshot["p0_value"], start_snapshot["p0_target"],
                            start_snapshot["p0_scale"],
                        ) if mode != "q_only" else _nan_change_summary()
                    ),
                    "pi": (
                        _fit_error(
                            post_p_snapshot["pi_value"], start_snapshot["pi_target"],
                            start_snapshot["pi_scale"],
                        ) if mode != "q_only" else _nan_change_summary()
                    ),
                    "q": (
                        _q_fit_by_regime(
                            post_q_snapshot["q_value"], post_p_snapshot["q_target"],
                            post_p_snapshot["q_scale"], post_p_snapshot["q_regime"],
                        ) if mode != "p_only" else {
                            name: _nan_change_summary()
                            for name in ("overall", "zero", "default", "survival")
                        }
                    ),
                }
                if mode == "coupled":
                    cross_effects = {
                        "p_to_q": _target_difference(
                            post_p_snapshot["q_target"], start_snapshot["q_target"],
                            start_snapshot["q_scale"],
                        ),
                        "q_to_p0": _target_difference(
                            post_q_snapshot["p0_target"], post_p_snapshot["p0_target"],
                            post_p_snapshot["p0_scale"], include_signed=True,
                        ),
                        "q_to_pi": _target_difference(
                            post_q_snapshot["pi_target"], post_p_snapshot["pi_target"],
                            post_p_snapshot["pi_scale"], include_signed=True,
                        ),
                    }
                else:
                    cross_effects = {
                        name: _nan_change_summary()
                        for name in ("p_to_q", "q_to_p0", "q_to_pi")
                    }
                current_targets = {
                    "p0": start_snapshot["p0_target"],
                    "pi": start_snapshot["pi_target"],
                    "q": post_p_snapshot["q_target"],
                }
                target_movement = {
                    name: (
                        _target_difference(
                            current_targets[name], self.previous_targets[name],
                            (
                                start_snapshot[f"{name}_scale"]
                                if name in {"p0", "pi"}
                                else post_p_snapshot["q_scale"]
                            ),
                        ) if self.previous_targets is not None else _nan_change_summary()
                    )
                    for name in ("p0", "pi", "q")
                }
                q_training = _q_training_diagnostics(q_summary)
                eval_seconds = time.perf_counter() - eval_start
            except Exception as exc:
                numerical_failure = f"{type(exc).__name__}: {exc}"
                break

            invariant_checks = {
                "bp_hash_unchanged": hashes_post_q["bp"] == self.initial_component_hashes["bp"],
                "sdf_fc1_hash_unchanged": (
                    hashes_post_q["sdf_fc1"] == self.initial_component_hashes["sdf_fc1"]
                ),
                "q_hash_unchanged": (
                    hashes_post_q["q"] == self.initial_component_hashes["q"]
                    if mode == "p_only" else None
                ),
                "p_hash_unchanged": (
                    hashes_post_q["p"] == self.initial_component_hashes["p"]
                    if mode == "q_only" else None
                ),
                "q_only_default_tensor_unchanged": (
                    bool(torch.equal(post_q_boundary["default"], self.initial_boundary["default"]))
                    if mode == "q_only" and self.initial_boundary is not None else None
                ),
                "q_only_regime_counts_unchanged": (
                    operator["q_regime_counts"] == self.initial_q_regime_counts
                    if mode == "q_only" else None
                ),
            }
            failed_invariants = [
                key for key, value in invariant_checks.items() if value is False
            ]
            if failed_invariants:
                numerical_failure = "RuntimeError: block-isolation invariant failed: " + ", ".join(
                    failed_invariants
                )
                break

            active_functions = ("p0", "pi") if mode == "p_only" else (
                ("q",) if mode == "q_only" else ("p0", "pi", "q")
            )
            finite_changes = all(
                math.isfinite(changes[name]["normalized_mean"])
                and math.isfinite(changes[name]["normalized_p90"])
                for name in active_functions
            )
            function_pass = bool(
                finite_changes
                and all(
                    changes[name]["normalized_mean"] < self.config.change_mean_tol
                    and changes[name]["normalized_p90"] < self.config.change_p90_tol
                    for name in active_functions
                )
            )
            boundary_pass = (
                bool(invariant_checks["q_only_default_tensor_unchanged"])
                if mode == "q_only" else bool(
                    math.isfinite(boundary_summary["default_flip_share"])
                    and boundary_summary["default_flip_share"] < self.config.default_flip_tol
                )
            )
            policy_pass = True if mode == "q_only" else bool(all(
                    math.isfinite(policy_changes[name]["p90"])
                    and policy_changes[name]["p90"] < self.config.bstar_p90_tol
                    for name in ("b0_star", "bi_star")
                ))
            function_two_cycle = False
            q_two_cycle = False
            if self.previous_previous_values is not None:
                delta_two_by_function = {
                    name: float(
                        (values[name] - self.previous_previous_values[name])
                        .abs().quantile(0.9).item()
                    )
                    for name in active_functions
                }
                function_two_cycle = bool(
                    max(delta_two_by_function.values()) < self.config.change_p90_tol
                    and max(changes[name]["p90"] for name in active_functions)
                    > 5.0 * self.config.change_p90_tol
                )
                q_two_cycle = bool(mode != "p_only" and
                    delta_two_by_function["q"] < self.config.change_p90_tol
                    and changes["q"]["p90"] > 5.0 * self.config.change_p90_tol
                )
            default_two_cycle = False
            if self.previous_previous_boundary is not None:
                default_flip_two = float(
                    (boundary["default"] != self.previous_previous_boundary["default"])
                    .float().mean().item()
                )
                default_two_cycle = bool(
                    default_flip_two < self.config.default_flip_tol
                    and boundary_summary["default_flip_share"]
                    > 5.0 * self.config.default_flip_tol
                )
            two_cycle = function_two_cycle or q_two_cycle or default_two_cycle
            recent = self.history[-4:]
            active_equations = ("p0", "pi") if mode == "p_only" else (
                ("q",) if mode == "q_only" else ("p0", "pi", "q")
            )
            primary_mean = max(
                operator["equations"][name]["normalized"]["mean"]
                for name in active_equations
            )
            primary_p90 = max(
                operator["equations"][name]["normalized"]["p90"]
                for name in active_equations
            )
            residual_stalled = bool(
                len(recent) == 4
                and primary_p90
                >= 0.99 * min(
                    row["primary_operator"]["normalized_p90"]
                    for row in recent
                )
            )
            function_stalled = bool(
                len(recent) == 4
                and max(changes[name]["normalized_p90"] for name in active_functions)
                >= 0.99 * min(
                    max(row["function_change"][name]["normalized_p90"] for name in active_functions)
                    for row in recent
                )
            )
            gates = {
                "function_change_passed": function_pass,
                "default_boundary_passed": boundary_pass,
                "implicit_policy_passed": policy_pass,
                "two_cycle_detected": two_cycle,
                "function_two_cycle_detected": function_two_cycle,
                "q_surface_two_cycle_detected": q_two_cycle,
                "default_boundary_two_cycle_detected": default_two_cycle,
                "residual_stalled_last_5": residual_stalled,
                "function_change_stalled_last_5": function_stalled,
            }
            sweep_pass = bool(
                sweep >= self.config.min_sweeps
                and operator["bellman_passed"] is True
                and function_pass and boundary_pass and policy_pass and not two_cycle
            )
            self.consecutive_passes = self.consecutive_passes + 1 if sweep_pass else 0
            row = {
                "mode": mode,
                "sweep": int(sweep),
                "operator": operator,
                "operator_stages": {
                    "start": start_operator,
                    "post_p": post_p_operator,
                    "post_q": operator,
                },
                "primary_operator": {
                    "equations": list(active_equations),
                    "normalized_mean": float(primary_mean),
                    "normalized_p90": float(primary_p90),
                },
                "function_change": changes,
                "frozen_target_fit": fit,
                "cross_effects": cross_effects,
                "target_movement": target_movement,
                "default_boundary": boundary_summary,
                "default_boundary_stages": {
                    "start": boundary_metrics(start_boundary, None),
                    "post_p": boundary_metrics(post_p_boundary, start_boundary),
                    "post_q": boundary_metrics(post_q_boundary, post_p_boundary),
                },
                "implicit_policy": policy_changes,
                "q_training_diagnostics": q_training,
                "invariants": invariant_checks,
                "gates": gates,
                "sweep_passed": sweep_pass,
                "consecutive_passes": int(self.consecutive_passes),
                "hashes": {
                    "start": hashes_start,
                    "post_p": hashes_post_p,
                    "post_q": hashes_post_q,
                },
                "training": {
                    "p": p_summary,
                    "q": q_summary,
                },
                "timing": {
                    "p_stage_seconds": p_seconds,
                    "q_stage_seconds": q_seconds,
                    "evaluation_seconds": eval_seconds,
                    "total_sweep_seconds": time.perf_counter() - sweep_start,
                },
            }
            self.history.append(row)
            boundary_dir = self.output_dir / "fixed_point" / "boundaries"
            boundary_dir.mkdir(parents=True, exist_ok=True)
            torch.save(boundary, boundary_dir / f"sweep_{sweep:03d}.pt")
            surface_dir = self.output_dir / "fixed_point" / "surfaces"
            surface_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "values": values,
                "policies": policies,
                "operator_snapshots": {
                    "start": start_snapshot,
                    "post_p": post_p_snapshot,
                    "post_q": post_q_snapshot,
                },
            }, surface_dir / f"sweep_{sweep:03d}.pt")
            self.previous_previous_values = self.previous_values
            self.previous_values = values
            self.previous_previous_boundary = self.previous_boundary
            self.previous_boundary = boundary
            self.previous_policies = policies
            self.previous_targets = current_targets
            self._save_checkpoint(
                sweep, f"sweep_{sweep:03d}.pt", {"stage": "post_q", "metrics": row}
            )
            self._write_metrics()
            if self.consecutive_passes >= self.config.consecutive_passes:
                passed = True
                first_pass_sweep = sweep - self.config.consecutive_passes + 1
                break

        if numerical_failure is not None:
            failure_reasons = ["numerical_failure"]
        elif passed:
            failure_reasons = []
        elif self.history:
            failure_reasons = self._failure_reasons(self.history[-1])
        else:
            failure_reasons = ["numerical_failure"]
        latest = self.history[-1] if self.history else None
        summary = {
            "mode": mode,
            "inner_fp_passed": bool(passed),
            "first_pass_sweep": first_pass_sweep,
            "completed_sweeps": len(self.history),
            "failure_reason": failure_reasons,
            "numerical_failure": numerical_failure,
            "statement": (
                "From the ep1 initialization, a numerical inner fixed point was found "
                "on the frozen ep1 Macro Law and fixed collocation/shock banks."
                if passed else
                "The solver ran, but the convergence gate is unconfigured because the "
                "conditional Bellman thresholds are None; no convergence verdict is made."
                if latest is not None and not latest["operator"].get("bellman_gate_configured", False) else
                "No numerical inner fixed point was found under the configured frozen "
                "Macro Law, collocation bank, shock bank, and acceptance thresholds."
            ),
            "uniqueness_claimed": False,
            "final_empirical_fp_distance_mean": (
                latest["operator"]["empirical_fixed_point_distance_mean"] if latest else None
            ),
            "final_empirical_fp_distance_p90": (
                latest["operator"]["empirical_fixed_point_distance_p90"] if latest else None
            ),
            "final_operator": latest["operator"] if latest else None,
            "final_function_change": latest["function_change"] if latest else None,
            "final_default_boundary": latest["default_boundary"] if latest else None,
            "final_implicit_policy": latest["implicit_policy"] if latest else None,
            "thresholds": asdict(self.config),
            "frozen_sdf_fc1_hash": self.sdf_hash,
            "bank_hashes": self.bank_hashes,
            "initial_component_hashes": self.initial_component_hashes,
            "final_invariants": latest["invariants"] if latest else None,
            "source_provenance": self.source_provenance,
        }
        self._write_metrics()
        write_json(self.output_dir / "fixed_point" / "final_summary.json", summary)
        if self.history:
            self._save_checkpoint(
                int(self.history[-1]["sweep"]), "final.pt", {"stage": "final", "final_summary": summary}
            )
        return summary
