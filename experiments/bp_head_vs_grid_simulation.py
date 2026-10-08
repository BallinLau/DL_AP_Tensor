"""Paired, read-only simulation experiment for BP-head versus grid actions.

The formal simulator remains the data-generating process in both arms.  The
GRID arm changes only the leverage action returned at a processed node; its
children, SDF values, and Bellman objective are built by the existing
``BPGridTeacher`` and convergence-transition helpers.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.checkpoint_loader import load_analysis_checkpoint  # noqa: E402
from data.simulate_ts import SimulateTS  # noqa: E402
from evaluation.bellman_diagnostics import evaluate_bellman_residuals  # noqa: E402
from evaluation.bp_diagnostics import (  # noqa: E402
    FrozenTransitionData,
    _checkpoint_economic_config,
    _losses,
    build_frozen_transition_data,
)
from evaluation.full_run_diagnostics import model_state_hash  # noqa: E402
from evaluation.grids import FrozenFirmGrid, ReferenceFirmState  # noqa: E402
from losses.q_loss import resolve_q_regime_settings  # noqa: E402
from training.bp_grid_teacher import BPGridTeacher  # noqa: E402
from training.bp_simulation_policy import (  # noqa: E402
    GridBPSimulationPolicy,
    compose_grid_simulation_action,
)
from training.episode import Episode  # noqa: E402


EPS = 1e-12
STATE_COLUMNS = ("b", "z", "ETA", "i", "x", "Hatcf", "LnKF", "M")


@dataclass
class RNGSnapshot:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor] | None


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def capture_rng_state() -> RNGSnapshot:
    return RNGSnapshot(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=(
            [value.clone() for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    )


def restore_rng_state(snapshot: RNGSnapshot) -> None:
    random.setstate(snapshot.python)
    np.random.set_state(snapshot.numpy)
    torch.set_rng_state(snapshot.torch_cpu)
    if snapshot.torch_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot.torch_cuda)


def rng_state_equal(left: RNGSnapshot, right: RNGSnapshot) -> bool:
    if left.python != right.python:
        return False
    if left.numpy[0] != right.numpy[0] or not np.array_equal(left.numpy[1], right.numpy[1]):
        return False
    if left.numpy[2:] != right.numpy[2:]:
        return False
    if not torch.equal(left.torch_cpu, right.torch_cpu):
        return False
    if (left.torch_cuda is None) != (right.torch_cuda is None):
        return False
    if left.torch_cuda is not None:
        return all(torch.equal(a, b) for a, b in zip(left.torch_cuda, right.torch_cuda))
    return True


def compose_grid_action(
    p0_star: torch.Tensor,
    mix_star: torch.Tensor,
    survival_probability: torch.Tensor,
) -> torch.Tensor:
    """Backward-compatible experiment alias for the production helper."""
    return compose_grid_simulation_action(
        p0_star,
        mix_star,
        survival_probability,
    )


class GridPolicyOverride:
    """RNG-neutral production-grid action evaluator used only by this experiment."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        sdf_fc1: torch.nn.Module,
        hyperparams: Any,
        economic_config: Any,
        n_child_shocks: int,
        shock_seed: int,
        record_rows: bool,
    ) -> None:
        self.model = model
        self.sdf_fc1 = sdf_fc1
        self.hyperparams = hyperparams
        self.economic_config = economic_config
        self.n_child_shocks = int(n_child_shocks)
        self.shock_seed = int(shock_seed)
        self.record_rows = bool(record_rows)
        p0_loss, pi_loss = _losses(economic_config)
        self.teacher = BPGridTeacher.from_hyperparams(
            model, p0_loss, pi_loss, hyperparams
        )
        self.reference = ReferenceFirmState(
            eta=1.0,
            i_low=0.0,
            i_mid=float(economic_config.I_THRESHOLD) / 2.0,
            i_high=float(economic_config.I_THRESHOLD),
            x=0.0,
            hatcf=0.0,
            lnkf=0.0,
            hatc_cal=0.0,
            lnk_cal=0.0,
            n_parent_rows=0,
            source="runtime_simulation",
            macro_source="runtime_simulation",
        )
        self.records: list[pd.DataFrame] = []
        self.calls = 0
        self.rows = 0
        self.active_rows = 0
        self.coarse_candidate_evaluations = 0
        self.fine_candidate_evaluations = 0

    @torch.no_grad()
    def evaluate_actions(
        self,
        *,
        firm_state: torch.Tensor,
        model_output: Any | None,
        hatc_cal: torch.Tensor,
        lnk_cal: torch.Tensor,
        include_pi: bool,
    ) -> Dict[str, torch.Tensor]:
        output = model_output if model_output is not None else self.model(firm_state)
        transition = build_frozen_transition_data(
            self.sdf_fc1,
            firm_state,
            self.reference,
            self.hyperparams,
            self.economic_config,
            n_child_shocks=self.n_child_shocks,
            shock_seed=self.shock_seed,
            hatc_cal_values=hatc_cal,
            lnk_cal_values=lnk_cal,
        )
        children = transition.children
        m_list = transition.m_used_list
        weights = transition.branch_weights
        p0 = self.teacher.compute(
            firm_state,
            children,
            m_list,
            branch="p0",
            bp_pred=output.bp0,
            child_weights=weights,
        )
        mix_weight = output.bar_i_cond.detach()
        mix_pred = output.bp_cond.detach()
        mixed = self.teacher.compute(
            firm_state,
            children,
            m_list,
            branch="mix",
            bp_pred=mix_pred,
            mix_weight=mix_weight,
            child_weights=weights,
        )
        pi = None
        if include_pi:
            pi = self.teacher.compute(
                firm_state,
                children,
                m_list,
                branch="pi",
                bp_pred=output.bpI,
                child_weights=weights,
            )
        action = compose_grid_action(
            p0["bp_star"], mixed["bp_star"], output.survival_prob
        )
        result = {
            "bp_head": output.bp.detach(),
            "bp_grid": action.detach(),
            "bp0_head": output.bp0.detach(),
            "bp0_grid": p0["bp_star"].detach(),
            "bpi_head": output.bpI.detach(),
            "bpi_grid": (
                pi["bp_star"].detach() if pi is not None else torch.full_like(action, float("nan"))
            ),
            "mix_head": mix_pred,
            "mix_grid": mixed["bp_star"].detach(),
            "survival_probability": output.survival_prob.detach(),
            "phat": output.Phat.detach(),
            "p_value": output.P.detach(),
            "top2_margin": mixed["top2_margin"].detach(),
            "regret": mixed["regret"].detach(),
            "refi_active": mixed["refi_active"].detach(),
        }
        return result

    @torch.no_grad()
    def __call__(self, **context: Any) -> torch.Tensor:
        # Only parent-node actions feed the next leverage transition. Branch
        # rows are reporting nodes and are re-evaluated as a parent before any
        # subsequent transition, so optimizing them here would add cost without
        # changing the simulated state path.
        if int(context["branch"]) != -1:
            return context["bp_head"]
        result = self.evaluate_actions(
            firm_state=context["firm_state"],
            model_output=context["model_output"],
            hatc_cal=context["hatc_cal"],
            lnk_cal=context["lnk_cal"],
            include_pi=False,
        )
        n = int(context["firm_state"].shape[0])
        active = int((context["firm_state"][:, 2] > 0.5).sum().item())
        self.calls += 1
        self.rows += n
        self.active_rows += active
        self.coarse_candidate_evaluations += 2 * active * int(self.teacher.coarse_size)
        if self.teacher.refine:
            self.fine_candidate_evaluations += 2 * active * int(self.teacher.fine_size)
        if self.record_rows:
            frame = pd.DataFrame({
                "path": context["path_index"].detach().cpu().numpy().astype(np.int64),
                "t": np.repeat(int(context["t"]), n),
                "branch": np.repeat(int(context["branch"]), n),
                "ID": context["firm_id"].detach().cpu().numpy().astype(np.int64),
            })
            for name, value in result.items():
                frame[name] = value.detach().cpu().reshape(-1).numpy()
            self.records.append(frame)
        return result["bp_grid"]

    def instrumentation(self) -> Dict[str, Any]:
        return {
            "override_calls": int(self.calls),
            "override_rows": int(self.rows),
            "refinancing_active_rows": int(self.active_rows),
            "coarse_candidate_evaluations": int(self.coarse_candidate_evaluations),
            "fine_candidate_evaluations": int(self.fine_candidate_evaluations),
            **self.teacher.forward_stats(),
        }


def discover_checkpoint(checkpoint_dir: Path, episode: int) -> Path:
    candidates = [
        checkpoint_dir / f"ep{episode}_combined.pt",
        checkpoint_dir.parent / "checkpoints_analysis" / f"ep{episode}_combined.pt",
        checkpoint_dir.parent / "checkpoints" / f"ep{episode}_combined.pt",
    ]
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        raise FileNotFoundError(f"No ep{episode}_combined.pt found; tried: {candidates}")
    return found.resolve()


def checkpoint_config_class(economic_config: Any) -> type:
    from config import Config

    return type(
        "CheckpointSimulationConfig",
        (Config,),
        dict(economic_config.to_dict()),
    )


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_simulation_arm(
    *,
    models: Mapping[str, torch.nn.Module | None],
    config_class: type,
    device: torch.device,
    n_paths: int,
    group_size: int,
    horizon: int,
    branch_num: int,
    enable_entry: bool,
    enable_exit: bool,
    grid_policy: GridBPSimulationPolicy | None,
) -> tuple[Any, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = int(torch.cuda.memory_allocated(device))
    else:
        baseline_memory = 0
    sim = SimulateTS(
        models=dict(models),
        config=config_class,
        n_paths=int(n_paths),
        group_size=int(group_size),
        horizon=int(horizon),
        branch_num=int(branch_num),
        main_branch=0,
        enable_entry=bool(enable_entry),
        enable_exit=bool(enable_exit),
        device=device,
        bp_action_source="grid" if grid_policy is not None else "head",
        bp_grid_policy=grid_policy,
    )
    _cuda_sync(device)
    started = time.perf_counter()
    tensor_output = sim.simulate_tensor()
    _cuda_sync(device)
    elapsed = time.perf_counter() - started
    if device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = 0
    firm, macro = tensor_output.to_dataframes()
    parent_steps = int((pd.to_numeric(firm["branch"], errors="coerce") == -1).sum())
    metrics = {
        "seconds": float(elapsed),
        "parent_firm_steps": parent_steps,
        "seconds_per_parent_firm_step": float(elapsed / max(parent_steps, 1)),
        "peak_gpu_mb": float(peak / 2**20),
        "peak_incremental_gpu_mb": float(max(peak - baseline_memory, 0) / 2**20),
    }
    return tensor_output, firm, macro, metrics


def attach_macro_state(firm: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    parents = firm[pd.to_numeric(firm["branch"], errors="coerce") == -1].copy()
    macro_parent = macro[pd.to_numeric(macro["branch"], errors="coerce") == -1].copy()
    calculated = macro_parent[["path", "t", "branch", "Hatc", "LnK"]].rename(
        columns={"Hatc": "hatc_cal", "LnK": "lnk_cal"}
    )
    merged = parents.merge(calculated, on=["path", "t", "branch"], how="left", validate="many_to_one")
    if merged[["hatc_cal", "lnk_cal"]].isna().any().any():
        raise RuntimeError("Could not attach calculated macro state to all parent rows")
    return merged


@torch.no_grad()
def evaluate_head_parent_actions(
    evaluator: GridPolicyOverride,
    parents: pd.DataFrame,
    *,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for start in range(0, len(parents), max(1, int(batch_size))):
        frame = parents.iloc[start:start + max(1, int(batch_size))].copy()
        state = torch.as_tensor(
            frame[["b", "z", "ETA", "i", "x", "Hatcf", "LnKF"]].to_numpy(np.float32),
            device=device,
        )
        result = evaluator.evaluate_actions(
            firm_state=state,
            model_output=None,
            hatc_cal=torch.as_tensor(frame["hatc_cal"].to_numpy(np.float32), device=device).reshape(-1, 1),
            lnk_cal=torch.as_tensor(frame["lnk_cal"].to_numpy(np.float32), device=device).reshape(-1, 1),
            include_pi=True,
        )
        keep = frame[["path", "t", "branch", "ID", "b", "z", "ETA", "i"]].reset_index(drop=True)
        for name, value in result.items():
            keep[name] = value.detach().cpu().reshape(-1).numpy()
        chunks.append(keep)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def one_dimensional_wasserstein(left: Iterable[float], right: Iterable[float]) -> float:
    a = np.asarray(list(left), dtype=np.float64)
    b = np.asarray(list(right), dtype=np.float64)
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    if a.size == 0 or b.size == 0:
        return float("nan")
    points = np.sort(np.concatenate([a, b]))
    if points.size < 2:
        return 0.0
    cdf_a = np.searchsorted(a, points[:-1], side="right") / float(a.size)
    cdf_b = np.searchsorted(b, points[:-1], side="right") / float(b.size)
    return float(np.sum(np.abs(cdf_a - cdf_b) * np.diff(points)))


def one_dimensional_ks(left: Iterable[float], right: Iterable[float]) -> float:
    a = np.sort(np.asarray(list(left), dtype=np.float64))
    b = np.sort(np.asarray(list(right), dtype=np.float64))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    points = np.sort(np.concatenate([a, b]))
    return float(np.max(np.abs(
        np.searchsorted(a, points, side="right") / float(a.size)
        - np.searchsorted(b, points, side="right") / float(b.size)
    )))


def _pre_exit_frame(
    full_firm: pd.DataFrame,
    parent_frame: pd.DataFrame,
    t: int,
) -> pd.DataFrame:
    """Return the main-child state before exit, or the initial parent at t=0."""
    if int(t) == 0:
        return parent_frame
    branch = pd.to_numeric(full_firm["branch"], errors="coerce")
    time_index = pd.to_numeric(full_firm["t"], errors="coerce").astype(int)
    return full_firm[(branch == 0) & (time_index == int(t))]


def distribution_by_t(
    head: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    head_full: pd.DataFrame,
    grid_full: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    times = sorted(set(head["t"].astype(int)).union(set(grid["t"].astype(int))))
    for t in times:
        h = head[head["t"].astype(int) == t]
        g = grid[grid["t"].astype(int) == t]
        pre_exit = {
            "head": _pre_exit_frame(head_full, h, int(t)),
            "grid": _pre_exit_frame(grid_full, g, int(t)),
        }
        merged = h[["path", "t", "ID", "b"]].merge(
            g[["path", "t", "ID", "b"]], on=["path", "t", "ID"], suffixes=("_head", "_grid")
        )
        row: Dict[str, Any] = {"t": int(t)}
        for label, frame in (("head", h), ("grid", g)):
            event_frame = pre_exit[label]
            b = pd.to_numeric(frame["b"], errors="coerce").to_numpy(np.float64)
            z = pd.to_numeric(frame["z"], errors="coerce").to_numpy(np.float64)
            row.update({
                f"firm_count_{label}": int(len(frame)),
                f"pre_exit_firm_count_{label}": int(len(event_frame)),
                f"survival_rate_{label}": float(
                    (pd.to_numeric(event_frame["P"], errors="coerce") > 0.0).mean()
                ),
                f"default_rate_{label}": float(
                    (pd.to_numeric(event_frame["Bar_z"], errors="coerce") >= 0.5).mean()
                ),
                f"entry_rate_{label}": float(
                    (pd.to_numeric(event_frame["entry"], errors="coerce") > 0.5).mean()
                ),
                f"mean_b_{label}": float(np.nanmean(b)),
                f"b_p10_{label}": float(np.nanquantile(b, 0.10)),
                f"b_p50_{label}": float(np.nanquantile(b, 0.50)),
                f"b_p90_{label}": float(np.nanquantile(b, 0.90)),
                f"b_p99_{label}": float(np.nanquantile(b, 0.99)),
                f"mean_z_{label}": float(np.nanmean(z)),
                f"z_p10_{label}": float(np.nanquantile(z, 0.10)),
                f"z_p50_{label}": float(np.nanquantile(z, 0.50)),
                f"z_p90_{label}": float(np.nanquantile(z, 0.90)),
                f"investment_rate_{label}": float((pd.to_numeric(frame["Bar_i"], errors="coerce") >= 0.5).mean()),
                f"mean_capital_{label}": float(pd.to_numeric(frame["K"], errors="coerce").mean()),
            })
        row["W1_b"] = one_dimensional_wasserstein(h["b"], g["b"])
        row["KS_b"] = one_dimensional_ks(h["b"], g["b"])
        row["firm_count_relative_gap"] = abs(len(h) - len(g)) / float(max(len(h), len(g), 1))
        row["matched_incumbent_n"] = int(len(merged))
        row["MAE_b_matched"] = (
            float(np.mean(np.abs(merged["b_head"] - merged["b_grid"])))
            if len(merged)
            else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def validate_paired_random_inputs(head: pd.DataFrame, grid: pd.DataFrame, *, atol: float = 1e-7) -> Dict[str, Any]:
    keys = ["path", "t", "ID"]
    columns = ["x", "z", "ETA", "i", "entry"]
    merged = head[keys + columns].merge(grid[keys + columns], on=keys, suffixes=("_head", "_grid"))
    errors: Dict[str, float] = {}
    for name in columns:
        delta = np.abs(
            pd.to_numeric(merged[f"{name}_head"], errors="coerce").to_numpy(np.float64)
            - pd.to_numeric(merged[f"{name}_grid"], errors="coerce").to_numpy(np.float64)
        )
        errors[name] = float(np.nanmax(delta)) if delta.size else float("nan")
    finite_errors = [value for value in errors.values() if np.isfinite(value)]
    passed = bool(finite_errors) and max(finite_errors) <= float(atol)
    if not passed:
        raise RuntimeError(f"HEAD/GRID random-input replay mismatch: {errors}")
    return {"common_rows": int(len(merged)), "max_abs_error_by_field": errors, "passed": True}


def summarize_bp_fit(state_level: pd.DataFrame) -> pd.DataFrame:
    data = state_level.copy()
    data["gap"] = data["bp_head"] - data["bp_grid"]
    data["abs_gap"] = data["gap"].abs()
    data["survival_region"] = np.where(data["phat"] > 0.0, "survival", "default")
    data["b_bin"] = pd.cut(data["b"], [-np.inf, 0.1, 0.3, 0.5, 0.7, 0.9, np.inf]).astype(str)
    data["z_bin"] = pd.cut(data["z"], [-np.inf, -2.0, 0.0, 2.0, np.inf]).astype(str)
    specs = [("overall", pd.Series(True, index=data.index))]
    for name in ("ETA", "b_bin", "z_bin", "survival_region"):
        for value in data[name].drop_duplicates():
            specs.append((f"{name}={value}", data[name] == value))
    rows = []
    for label, mask in specs:
        sample = data[mask & np.isfinite(data["abs_gap"])]
        if sample.empty:
            continue
        gap = sample["gap"].to_numpy(np.float64)
        absolute = np.abs(gap)
        rows.append({
            "group": label,
            "n": int(len(sample)),
            "signed_bias": float(gap.mean()),
            "mae": float(absolute.mean()),
            "p50_abs_gap": float(np.quantile(absolute, 0.50)),
            "p90_abs_gap": float(np.quantile(absolute, 0.90)),
            "p99_abs_gap": float(np.quantile(absolute, 0.99)),
            "pr_head_gt_grid": float((gap > 0.0).mean()),
            "pr_head_lt_grid": float((gap < 0.0).mean()),
        })
    return pd.DataFrame(rows)


def _quantile_row(arm: str, metric: str, values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"arm": arm, "metric": metric, "n": 0, "mean": np.nan, "p50": np.nan, "p90": np.nan, "p99": np.nan}
    return {
        "arm": arm,
        "metric": metric,
        "n": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
    }


@torch.no_grad()
def evaluate_pq_target_bank(
    *,
    arm: str,
    tensor_table: Any,
    model: torch.nn.Module,
    sdf_fc1: torch.nn.Module,
    hyperparams: Any,
    economic_config: Any,
    config_class: type,
    device: torch.device,
    batch_size: int,
    branch_num: int,
    q_settings: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = deepcopy(model).to(device).eval().requires_grad_(False)
    episode = Episode(
        models={"policy_value": model, "sdf_fc1": sdf_fc1},
        optimizers={},
        config=config_class,
        hyperparams=hyperparams,
        device=device,
        episode_id=1,
        firm_target=target,
        q_checkpoint_loaded=True,
    )
    pool = episode._tensor_to_parent_group_pool(
        tensor_table, n_branches=int(branch_num), source_id=0
    )
    batches = episode._parent_group_pool_to_batches(
        pool, batch_size=int(batch_size), eta_resample=False, shuffle=False
    )
    if not batches:
        raise RuntimeError(f"{arm} simulation produced no complete P/Q parent-child batches")
    pq_cache, _ = episode._build_pq_value_target_cache(
        batches, target, q_target_model=target
    )
    bp_cache = episode._build_bp_target_cache(batches, target)
    metrics: Dict[str, list[np.ndarray]] = {
        "P0_target": [], "PI_target": [], "Q_target_training": [],
        "Q_claim": [], "Q_effective": [], "Q_training_residual": [],
        "bp0_teacher": [], "bpi_teacher": [], "bp_mix_teacher_conditional": [],
        "bp_teacher": [],
        **{f"state_{name}": [] for name in STATE_COLUMNS},
    }
    for batch, pq_item, bp_item in zip(batches, pq_cache, bp_cache):
        parent = batch["parent"]
        metrics["P0_target"].append(pq_item.p0_value_target.detach().cpu().numpy().reshape(-1))
        metrics["PI_target"].append(pq_item.pi_value_target.detach().cpu().numpy().reshape(-1))
        metrics["bp0_teacher"].append(bp_item["bp0_target"].numpy().reshape(-1))
        metrics["bpi_teacher"].append(bp_item["bpi_target"].numpy().reshape(-1))
        bp0_target = bp_item["bp0_target"].numpy().reshape(-1)
        mix_target = bp_item["mix_target"].numpy().reshape(-1)
        survival = bp_item["mix_sample_weight"].numpy().reshape(-1)
        metrics["bp_mix_teacher_conditional"].append(mix_target)
        metrics["bp_teacher"].append(
            survival * mix_target + (1.0 - survival) * bp0_target
        )
        for idx, name in enumerate(STATE_COLUMNS):
            if idx < parent.shape[1]:
                metrics[f"state_{name}"].append(parent[:, idx].detach().cpu().numpy().reshape(-1))

        raw_m, used_m = episode._build_policy_m_lists(
            parent,
            batch["children"],
            float(getattr(hyperparams, "pv_m_clamp_min", 0.7)),
            float(getattr(hyperparams, "pv_m_clamp_max", 1.3)),
        )
        children, raw_expanded, used_expanded, weights = episode._expand_policy_expectation_children(
            batch["children"], raw_m, used_m
        )
        n = int(parent.shape[0])
        grid = FrozenFirmGrid(
            b_values=parent[:, 0].detach().cpu().numpy(),
            z_values=np.asarray([0.0]),
            mesh_b=parent[:, 0].detach().cpu().numpy().reshape(n, 1),
            mesh_z=parent[:, 1].detach().cpu().numpy().reshape(n, 1),
            base_states=parent[:, :7],
        )
        transition = FrozenTransitionData(
            children=list(children),
            m_raw_list=list(raw_expanded),
            m_used_list=list(used_expanded),
            branch_weights=weights,
            metadata={
                "source": "formal_simulation_parent_child_batches",
                "expanded_child_count": len(children),
            },
        )
        surfaces, _ = evaluate_bellman_residuals(
            target,
            grid,
            transition,
            economic_config,
            recovery_normalization_mode=q_settings["recovery_normalization_mode"],
            parent_default_regime_mode=q_settings["q_parent_default_regime_mode"],
            parent_default_eps=q_settings["q_parent_default_eps"],
            parent_default_tau=q_settings["q_parent_default_tau"],
        )
        metrics["Q_target_training"].append(np.asarray(surfaces["Q_target_training"]).reshape(-1))
        metrics["Q_claim"].append(np.asarray(surfaces["Q_claim"]).reshape(-1))
        metrics["Q_effective"].append(np.asarray(surfaces["Q_effective"]).reshape(-1))
        metrics["Q_training_residual"].append(
            np.asarray(surfaces["RQ_training_signed"]).reshape(-1)
        )

    with torch.no_grad():
        parent_all = torch.cat([batch["parent"][:, :7] for batch in batches], dim=0)
        phat = target.forward_equity(parent_all)["Phat"].detach().cpu().numpy().reshape(-1)
    metrics["survival_target_indicator"] = [(phat > 0.0).astype(float)]

    rows = []
    raw_columns: Dict[str, np.ndarray] = {}
    for metric, parts in metrics.items():
        values = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
        raw_columns[metric] = values
        rows.append(_quantile_row(arm, metric, values))
    raw = pd.DataFrame(raw_columns)
    return pd.DataFrame(rows), raw


def compare_target_banks(head_summary: pd.DataFrame, grid_summary: pd.DataFrame, head_raw: pd.DataFrame, grid_raw: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([head_summary, grid_summary], ignore_index=True)
    rows = []
    for metric in sorted(set(head_raw.columns).intersection(grid_raw.columns)):
        h = head_raw[metric].to_numpy(np.float64)
        g = grid_raw[metric].to_numpy(np.float64)
        h = h[np.isfinite(h)]; g = g[np.isfinite(g)]
        pooled_scale = float(np.std(np.concatenate([h, g]))) if h.size and g.size else float("nan")
        w1 = one_dimensional_wasserstein(h, g)
        rows.append({
            "arm": "HEAD_vs_GRID",
            "metric": metric,
            "n": int(min(h.size, g.size)),
            "mean": float(np.mean(h) - np.mean(g)) if h.size and g.size else np.nan,
            "p50": np.nan,
            "p90": w1,
            "p99": (w1 / max(pooled_scale, EPS)) if np.isfinite(w1) and np.isfinite(pooled_scale) else np.nan,
        })
    return pd.concat([combined, pd.DataFrame(rows)], ignore_index=True)


def _save_line_plot(frame: pd.DataFrame, y_head: str, y_grid: str, path: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frame["t"], frame[y_head], marker="o", label="HEAD")
    ax.plot(frame["t"], frame[y_grid], marker="o", label="GRID")
    ax.set(xlabel="t", ylabel=ylabel)
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def write_figures(output: Path, bp: pd.DataFrame, drift: pd.DataFrame, head_parent: pd.DataFrame, grid_parent: pd.DataFrame) -> None:
    figs = output / "figs"; figs.mkdir(parents=True, exist_ok=True)
    active = bp[(bp["ETA"] > 0.5) & np.isfinite(bp["bp_head"]) & np.isfinite(bp["bp_grid"])]
    fig, ax = plt.subplots(figsize=(5.5, 5.5)); ax.scatter(active["bp_grid"], active["bp_head"], s=8, alpha=0.35)
    ax.plot([0, 1], [0, 1], color="black", linewidth=1); ax.set(xlabel="grid BP", ylabel="head BP")
    fig.tight_layout(); fig.savefig(figs / "bp_head_vs_grid_scatter.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.hist(active["bp_head"] - active["bp_grid"], bins=50)
    ax.set(xlabel="bp_head - bp_grid", ylabel="count"); fig.tight_layout(); fig.savefig(figs / "bp_gap_hist.png", dpi=160); plt.close(fig)
    heat = active.assign(
        b_bin=pd.cut(active["b"], np.linspace(0, 1, 11), include_lowest=True),
        z_bin=pd.cut(active["z"], np.linspace(-4, 4, 17), include_lowest=True),
        abs_gap=(active["bp_head"] - active["bp_grid"]).abs(),
    ).pivot_table(index="z_bin", columns="b_bin", values="abs_gap", aggfunc="mean", observed=False)
    fig, ax = plt.subplots(figsize=(8, 5)); image = ax.imshow(heat.to_numpy(), origin="lower", aspect="auto")
    ax.set(xlabel="b bin", ylabel="z bin"); fig.colorbar(image, ax=ax, label="mean |BP gap|")
    fig.tight_layout(); fig.savefig(figs / "bp_gap_bz_heatmap.png", dpi=160); plt.close(fig)
    _save_line_plot(drift, "mean_b_head", "mean_b_grid", figs / "mean_b_head_vs_grid.png", "mean b")
    _save_line_plot(drift, "default_rate_head", "default_rate_grid", figs / "default_rate_head_vs_grid.png", "default rate")
    _save_line_plot(drift, "firm_count_head", "firm_count_grid", figs / "firm_count_head_vs_grid.png", "firm count")
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.plot(drift["t"], drift["W1_b"], marker="o", label="W1(b)")
    ax.plot(drift["t"], drift["MAE_b_matched"], marker="s", label="matched MAE(b)")
    ax.set(xlabel="t", ylabel="distance"); ax.legend(); fig.tight_layout(); fig.savefig(figs / "distribution_drift_b_over_time.png", dpi=160); plt.close(fig)
    final_t = int(max(head_parent["t"].max(), grid_parent["t"].max()))
    h = head_parent[head_parent["t"].astype(int) == final_t]; g = grid_parent[grid_parent["t"].astype(int) == final_t]
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.hist(h["b"], bins=40, alpha=0.5, density=True, label="HEAD"); ax.hist(g["b"], bins=40, alpha=0.5, density=True, label="GRID")
    ax.set(xlabel="b", ylabel="density"); ax.legend(); fig.tight_layout(); fig.savefig(figs / "b_hist_head_vs_grid.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    axes[0].scatter(h["b"], h["z"], s=5, alpha=0.25); axes[0].set_title("HEAD")
    axes[1].scatter(g["b"], g["z"], s=5, alpha=0.25); axes[1].set_title("GRID")
    for ax in axes: ax.set(xlabel="b", ylabel="z")
    fig.tight_layout(); fig.savefig(figs / "bz_scatter_or_density_head_vs_grid.png", dpi=160); plt.close(fig)


def classify_support(bp_summary: pd.DataFrame, drift: pd.DataFrame, pq: pd.DataFrame) -> tuple[str, Dict[str, Any]]:
    overall = bp_summary[bp_summary["group"] == "overall"].iloc[0]
    active_row = bp_summary[bp_summary["group"] == "ETA=1.0"]
    selected = active_row.iloc[0] if not active_row.empty else overall
    bp_material = bool(float(selected["mae"]) >= 0.03 or float(selected["p90_abs_gap"]) >= 0.05)
    max_w1 = float(drift["W1_b"].max())
    max_default_gap = float((drift["default_rate_head"] - drift["default_rate_grid"]).abs().max())
    max_firm_count_gap = float(drift["firm_count_relative_gap"].max())
    drift_material = bool(
        max_w1 >= 0.03
        or max_default_gap >= 0.01
        or max_firm_count_gap >= 0.05
    )
    impact = pq[(pq["arm"] == "HEAD_vs_GRID") & (pq["metric"].isin(["P0_target", "PI_target", "Q_target_training"]))]
    max_standardized_w1 = float(impact["p99"].max()) if not impact.empty else float("nan")
    target_material = bool(np.isfinite(max_standardized_w1) and max_standardized_w1 >= 0.10)
    if bp_material and drift_material and target_material:
        verdict = "Strong support"
    elif bp_material and drift_material:
        verdict = "Partial support"
    else:
        verdict = "Weak support"
    evidence = {
        "operational_thresholds": {
            "bp_mae": 0.03, "bp_p90_abs_gap": 0.05,
            "W1_b": 0.03, "default_rate_gap": 0.01,
            "firm_count_relative_gap": 0.05,
            "pq_target_standardized_W1": 0.10,
        },
        "bp_material": bp_material,
        "distribution_drift_material": drift_material,
        "pq_target_impact_material": target_material,
        "max_W1_b": max_w1,
        "max_default_rate_gap": max_default_gap,
        "max_firm_count_relative_gap": max_firm_count_gap,
        "max_pq_target_standardized_W1": max_standardized_w1,
    }
    return verdict, evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--n-paths", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--branch-num", type=int, default=2)
    parser.add_argument("--teacher-child-shocks", type=int, default=2)
    parser.add_argument("--teacher-shock-seed", type=int, default=12345)
    parser.add_argument("--simulation-seed", type=int, default=12345)
    parser.add_argument("--teacher-eval-batch-size", type=int, default=512)
    parser.add_argument("--target-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disable-entry", action="store_true")
    parser.add_argument("--disable-exit", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else discover_checkpoint(checkpoint_dir, args.episode)
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else checkpoint_dir.parent / "bp_head_vs_grid" / f"ep{args.episode}_n{args.n_paths}_g{args.group_size}_t{args.horizon}"
    )
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if int(args.branch_num) < 2:
        raise ValueError("branch-num must be at least 2 for Bellman target evaluation")
    if int(args.teacher_child_shocks) < 2:
        raise ValueError("teacher-child-shocks must be at least 2")

    loaded = load_analysis_checkpoint(checkpoint, device=device)
    model = loaded.models["policy_value"]
    sdf_fc1 = loaded.models["sdf_fc1"]
    if sdf_fc1 is None:
        raise ValueError("GRID simulation requires checkpoint SDF/FC1")
    if not bool(getattr(loaded.hyperparams, "pv_exact_eta_integration_enabled", True)):
        raise ValueError("This controlled experiment requires the formal exact-eta teacher semantics")
    for module in loaded.models.values():
        if isinstance(module, torch.nn.Module):
            module.eval(); module.requires_grad_(False)
    hashes_before = {
        name: model_state_hash(module)
        for name, module in loaded.models.items()
        if isinstance(module, torch.nn.Module)
    }
    config_class = checkpoint_config_class(loaded.economic_config)
    q_settings = resolve_q_regime_settings(
        checkpoint_recovery_normalization_mode=getattr(loaded.hyperparams, "q_recovery_normalization_mode", None),
        checkpoint_parent_default_regime_mode=getattr(loaded.hyperparams, "q_parent_default_regime_mode", None),
        checkpoint_parent_default_eps=getattr(loaded.hyperparams, "q_parent_default_eps", None),
        checkpoint_parent_default_tau=getattr(loaded.hyperparams, "q_parent_default_tau", None),
        checkpoint_recorded_fields=loaded.metadata.get("q_semantics_recorded_fields", []),
    )
    p0_loss, pi_loss = _losses(loaded.economic_config)
    grid_override = GridBPSimulationPolicy(
        target_model=model,
        sdf_fc1_model=sdf_fc1,
        p0_loss=p0_loss,
        pi_loss=pi_loss,
        hyperparams=loaded.hyperparams,
        economic_config=loaded.economic_config,
        n_child_shocks=args.teacher_child_shocks,
        shock_seed=args.teacher_shock_seed,
    )
    warmup_grid_override = GridBPSimulationPolicy(
        target_model=model,
        sdf_fc1_model=sdf_fc1,
        p0_loss=p0_loss,
        pi_loss=pi_loss,
        hyperparams=loaded.hyperparams,
        economic_config=loaded.economic_config,
        n_child_shocks=args.teacher_child_shocks,
        shock_seed=args.teacher_shock_seed,
    )

    seed_everything(args.simulation_seed)
    replay_state = capture_rng_state()
    with _checkpoint_economic_config(loaded.economic_config):
        # Warm both exact execution paths before timing so CUDA/kernel startup is
        # not charged only to HEAD. Warmup output and instrumentation are discarded.
        restore_rng_state(replay_state)
        run_simulation_arm(
            models=loaded.models, config_class=config_class, device=device,
            n_paths=args.n_paths, group_size=args.group_size, horizon=args.horizon,
            branch_num=args.branch_num, enable_entry=not args.disable_entry,
            enable_exit=not args.disable_exit, grid_policy=None,
        )
        restore_rng_state(replay_state)
        run_simulation_arm(
            models=loaded.models, config_class=config_class, device=device,
            n_paths=args.n_paths, group_size=args.group_size, horizon=args.horizon,
            branch_num=args.branch_num, enable_entry=not args.disable_entry,
            enable_exit=not args.disable_exit, grid_policy=warmup_grid_override,
        )

        # Both measured arms consume the exact same captured RNG state. The GRID
        # callback uses a dedicated CPU generator and cannot shift this tape.
        restore_rng_state(replay_state)
        head_output, head_firm, head_macro, head_runtime = run_simulation_arm(
            models=loaded.models, config_class=config_class, device=device,
            n_paths=args.n_paths, group_size=args.group_size, horizon=args.horizon,
            branch_num=args.branch_num, enable_entry=not args.disable_entry,
            enable_exit=not args.disable_exit, grid_policy=None,
        )
        head_final_rng = capture_rng_state()
        restore_rng_state(replay_state)
        grid_output, grid_firm, grid_macro, grid_runtime = run_simulation_arm(
            models=loaded.models, config_class=config_class, device=device,
            n_paths=args.n_paths, group_size=args.group_size, horizon=args.horizon,
            branch_num=args.branch_num, enable_entry=not args.disable_entry,
            enable_exit=not args.disable_exit, grid_policy=grid_override,
        )
        grid_final_rng = capture_rng_state()
        if not rng_state_equal(head_final_rng, grid_final_rng):
            raise RuntimeError(
                "HEAD and GRID consumed different simulation RNG tapes; "
                "the intervention is not strictly paired"
            )

        head_parent = attach_macro_state(head_firm, head_macro)
        grid_parent = attach_macro_state(grid_firm, grid_macro)
        pairing = validate_paired_random_inputs(head_parent, grid_parent)

        diagnostic_evaluator = GridPolicyOverride(
            model=model, sdf_fc1=sdf_fc1, hyperparams=loaded.hyperparams,
            economic_config=loaded.economic_config,
            n_child_shocks=args.teacher_child_shocks,
            shock_seed=args.teacher_shock_seed, record_rows=False,
        )
        rng_before_diagnostics = capture_rng_state()
        bp_state = evaluate_head_parent_actions(
            diagnostic_evaluator, head_parent, device=device,
            batch_size=args.teacher_eval_batch_size,
        )
        rng_after_diagnostics = capture_rng_state()
        if not rng_state_equal(rng_before_diagnostics, rng_after_diagnostics):
            raise RuntimeError("BP grid diagnostic consumed global RNG state")

        head_target_summary, head_target_raw = evaluate_pq_target_bank(
            arm="HEAD", tensor_table=head_output.firm, model=model, sdf_fc1=sdf_fc1,
            hyperparams=loaded.hyperparams, economic_config=loaded.economic_config,
            config_class=config_class, device=device, batch_size=args.target_batch_size,
            branch_num=args.branch_num, q_settings=q_settings,
        )
        grid_target_summary, grid_target_raw = evaluate_pq_target_bank(
            arm="GRID", tensor_table=grid_output.firm, model=model, sdf_fc1=sdf_fc1,
            hyperparams=loaded.hyperparams, economic_config=loaded.economic_config,
            config_class=config_class, device=device, batch_size=args.target_batch_size,
            branch_num=args.branch_num, q_settings=q_settings,
        )

    runtime = pd.DataFrame([
        {"setting": "HEAD", **head_runtime},
        {"setting": "GRID", **grid_runtime, **grid_override.instrumentation()},
    ])
    ratio = float(grid_runtime["seconds"] / max(head_runtime["seconds"], EPS))
    runtime["grid_over_head_runtime_ratio"] = ratio
    bp_summary = summarize_bp_fit(bp_state)
    drift = distribution_by_t(
        head_parent,
        grid_parent,
        head_full=head_firm,
        grid_full=grid_firm,
    )
    pq_summary = compare_target_banks(
        head_target_summary, grid_target_summary, head_target_raw, grid_target_raw
    )
    verdict, verdict_evidence = classify_support(bp_summary, drift, pq_summary)

    runtime.to_csv(output / "runtime_summary.csv", index=False)
    bp_summary.to_csv(output / "bp_fit_summary.csv", index=False)
    bp_state.to_csv(output / "bp_state_level.csv", index=False)
    drift.to_csv(output / "state_distribution_by_t.csv", index=False)
    pq_summary.to_csv(output / "pq_target_summary.csv", index=False)
    head_target_raw.to_csv(output / "pq_target_state_bank_head.csv", index=False)
    grid_target_raw.to_csv(output / "pq_target_state_bank_grid.csv", index=False)
    write_figures(output, bp_state, drift, head_parent, grid_parent)

    hashes_after = {
        name: model_state_hash(module)
        for name, module in loaded.models.items()
        if isinstance(module, torch.nn.Module)
    }
    if hashes_before != hashes_after:
        raise RuntimeError("Experiment modified frozen checkpoint parameters")
    config = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": loaded.metadata.get("checkpoint_sha256"),
        "episode": int(args.episode),
        "n_paths": int(args.n_paths),
        "group_size": int(args.group_size),
        "horizon": int(args.horizon),
        "branch_num": int(args.branch_num),
        "teacher_child_shocks": int(args.teacher_child_shocks),
        "teacher_shock_seed": int(args.teacher_shock_seed),
        "simulation_seed": int(args.simulation_seed),
        "paired_random_inputs": pairing,
        "paired_rng_final_state_equal": rng_state_equal(
            head_final_rng, grid_final_rng
        ),
        "formal_bp_action": "PolicyValueModel.forward_simulation().bp",
        "grid_bp_action": "survival_prob*mix_grid_star+(1-survival_prob)*p0_grid_star",
        "leverage_timing": "b_next=eta_current*bp+(1-eta_current)*b_current",
        "teacher_transition_builder": "GridBPSimulationPolicy/ConvergenceShockBank",
        "q_semantics": q_settings,
        "model_hashes_before": hashes_before,
        "model_hashes_after": hashes_after,
        "model_hash_invariant": hashes_before == hashes_after,
        "runtime_ratio": ratio,
        "conclusion": verdict,
        "conclusion_evidence": verdict_evidence,
        "second_stage_pq_training_run": False,
    }
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    report = f"""# BP Head vs Grid Simulation Experiment

## 1. Implementation audit
- Formal simulation BP: `PolicyValueModel.forward_simulation().bp`.
- GRID BP: production `BPGridTeacher` with checkpoint P/Q/SDF/FC1, exact eta integration, and current-parent eta timing.
- Episode>0 P/Q data: formal `SimulateTS` firm panel parsed through `Episode._tensor_to_parent_group_pool` and `_parent_group_pool_to_batches`.

## 2. Runtime
- Grid/Head wall-clock ratio: {ratio:.6g}.

## 3. BP fit
- See `bp_fit_summary.csv` and `bp_state_level.csv`.

## 4. Distribution drift
- Maximum W1(b): {verdict_evidence['max_W1_b']:.6g}.
- Maximum default-rate gap: {verdict_evidence['max_default_rate_gap']:.6g}.
- Maximum relative firm-count gap: {verdict_evidence['max_firm_count_relative_gap']:.6g}.

## 5. P/Q target impact
- Maximum standardized target W1: {verdict_evidence['max_pq_target_standardized_W1']:.6g}.
- No network update was run; all target comparisons use the same frozen checkpoint.

## 6. Conclusion
**{verdict}** under the operational thresholds recorded in `config.json`.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "conclusion": verdict, "runtime_ratio": ratio}, indent=2))


if __name__ == "__main__":
    main()
