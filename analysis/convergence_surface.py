from __future__ import annotations

import json
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from config import Config, HyperParams
from losses import P0Loss, PILoss, QLoss
from utils.firm_transition import apply_refinancing_policy

from .checkpoint_loader import AnalysisCheckpoint, load_analysis_checkpoint
from .convergence_transition import (
    ChildExogenousBundle,
    ConvergenceShockBank,
    FirmStateIndex,
    MacroTransitionContext,
    build_child_exogenous_bundle,
)


SURFACE_FAMILY = "bellman_fixed_point_conditional_mean_v1"
REQUIRED_FIXED_STATE = ("eta", "i", "x", "hatcf", "lnkf", "hatc_cal", "lnk_cal")


@dataclass
class ConvergenceSurfaceResult:
    long_table: pd.DataFrame
    raw_tensors: Dict[str, Any]
    manifests: Dict[str, Any]
    checkpoint_metadata: List[Dict[str, Any]]
    shock_bank_metadata: Dict[str, Any]
    support_metadata: Dict[str, Any]
    disabled_metrics: Dict[str, Any]


@contextmanager
def preserve_analysis_state(models: Iterable[torch.nn.Module]):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    snapshots = []
    try:
        for model in models:
            snapshots.append((
                model,
                model.training,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            ))
            model.eval()
        with torch.no_grad():
            yield
    finally:
        for model, was_training, state in snapshots:
            model.load_state_dict(state, strict=True)
            model.train(was_training)
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _policy_get(out: Any, name: str) -> torch.Tensor:
    if isinstance(out, dict):
        return out[name]
    return getattr(out, name)


def _train_m(eq: str, m_raw: torch.Tensor, hp: HyperParams) -> torch.Tensor:
    if eq in {"p0", "pi"}:
        if bool(getattr(hp, "pv_use_clipped_m", True)):
            return m_raw.clamp(
                float(getattr(hp, "pv_m_clamp_min", 0.7)),
                float(getattr(hp, "pv_m_clamp_max", 1.3)),
            )
        return m_raw
    if bool(getattr(hp, "q_use_detached_m", True)):
        return m_raw.detach().clamp(
            float(getattr(hp, "q_m_clamp_min", 0.5)),
            float(getattr(hp, "q_m_clamp_max", 1.5)),
        )
    return m_raw


def reduce_signed_surface(
    signed: torch.Tensor,
    weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    conditional_signed = (weights * signed).sum(dim=1)
    conditional_abs = conditional_signed.abs()
    realized_abs = (weights * signed.abs()).sum(dim=1)
    sum_w_r2 = (weights * signed.pow(2)).sum(dim=1)
    child_std = torch.sqrt((sum_w_r2 - conditional_signed.pow(2)).clamp_min(0.0))
    sum_w2 = weights.pow(2).sum(dim=1)
    denom = 1.0 - sum_w2
    eps = torch.finfo(weights.dtype).eps
    mc_se = child_std * torch.sqrt(sum_w2 / denom.clamp_min(eps))
    mc_se = torch.where(denom > eps, mc_se, torch.full_like(mc_se, float("nan")))
    return {
        "conditional_signed": conditional_signed,
        "conditional_abs": conditional_abs,
        "realized_abs": realized_abs,
        "child_dispersion": child_std,
        "mc_standard_error": mc_se,
    }


class VectorizedBellmanSurfaceBackend:
    def __init__(
        self,
        policy_model: torch.nn.Module,
        hyperparams: HyperParams,
        *,
        p0_loss: Optional[P0Loss] = None,
        pi_loss: Optional[PILoss] = None,
        q_loss: Optional[QLoss] = None,
    ) -> None:
        self.policy_model = policy_model
        self.hyperparams = hyperparams
        self.p0_loss = p0_loss or P0Loss()
        self.pi_loss = pi_loss or PILoss()
        self.q_loss = q_loss or QLoss()

    def compute_signed_residuals(
        self,
        parent_states: torch.Tensor,
        child: ChildExogenousBundle,
        *,
        equations: Sequence[str] = ("p0", "pi", "q"),
        m_modes: Sequence[str] = ("train", "raw"),
        child_chunk_size: int = 8192,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        idx = FirmStateIndex()
        bsz = parent_states.shape[0]
        n_child = child.z_next.shape[1]
        parent_out = self.policy_model(parent_states)
        bp0 = _policy_get(parent_out, "bp0")
        bpI = _policy_get(parent_out, "bpI")
        q_parent = _policy_get(parent_out, "Q")
        p0_parent = _policy_get(parent_out, "P0")
        pi_parent = _policy_get(parent_out, "PI")
        bar_i = _policy_get(parent_out, "bar_i")

        def make_child_states(b_child: torch.Tensor) -> torch.Tensor:
            return torch.stack([
                b_child,
                child.z_next.squeeze(-1),
                child.eta_next.squeeze(-1),
                child.i_next.squeeze(-1),
                child.x_next.squeeze(-1),
                child.hatcf_next.squeeze(-1),
                child.lnkf_next.squeeze(-1),
            ], dim=-1)

        eta = parent_states[:, idx.ETA:idx.ETA + 1].clamp(0.0, 1.0)
        b_parent = parent_states[:, idx.B:idx.B + 1]
        b_p0 = apply_refinancing_policy(b_current=b_parent, bp_candidate=bp0, eta_current=eta)
        b_pi = apply_refinancing_policy(b_current=b_parent, bp_candidate=bpI, eta_current=eta)
        multiplier = bar_i * (float(Config.G) - 1.0) + 1.0
        b_q = b_parent / multiplier.clamp_min(1e-6)

        child_states = {
            "p0": make_child_states(b_p0.expand(-1, n_child)),
            "pi": make_child_states(b_pi.expand(-1, n_child)),
            "q": make_child_states(b_q.expand(-1, n_child)),
        }

        outputs: Dict[str, Any] = {}
        for eq in equations:
            flat = child_states[eq].reshape(-1, 7)
            out_chunks = []
            step = max(1, int(child_chunk_size))
            for start in range(0, flat.shape[0], step):
                out_chunks.append(self.policy_model(flat[start:start + step]))

            def cat_field(name: str) -> torch.Tensor:
                vals = [_policy_get(out, name) for out in out_chunks]
                return torch.cat(vals, dim=0).reshape(bsz, n_child, 1)

            outputs[eq] = {
                "P": cat_field("P"),
                "bar_z": cat_field("bar_z"),
                "Q": cat_field("Q"),
            }

        q_exp = q_parent.unsqueeze(1).expand(-1, n_child, -1)
        p0_exp = p0_parent.unsqueeze(1).expand(-1, n_child, -1)
        pi_exp = pi_parent.unsqueeze(1).expand(-1, n_child, -1)
        m_raw = child.m_raw
        result: Dict[str, Dict[str, torch.Tensor]] = {}
        for eq in equations:
            m_by_mode = {
                "raw": m_raw,
                "train": _train_m(eq, m_raw, self.hyperparams),
            }
            result[eq] = {}
            if eq == "p0":
                childp0_state = parent_states.clone()
                childp0_state[:, idx.B:idx.B + 1] = b_p0
                q_p0 = _policy_get(self.policy_model(childp0_state), "Q")
                cf = self.p0_loss.compute_cashflow_p0(
                    parent_states[:, idx.X:idx.X + 1],
                    parent_states[:, idx.Z:idx.Z + 1],
                    b_parent,
                    q_parent,
                    q_p0,
                    eta,
                ).unsqueeze(1)
                for mode in m_modes:
                    result[eq][mode] = (
                        p0_exp - cf - m_by_mode[mode] * outputs[eq]["P"]
                    ).squeeze(-1)
            elif eq == "pi":
                childpi_state = parent_states.clone()
                childpi_state[:, idx.B:idx.B + 1] = b_pi
                q_pi = _policy_get(self.policy_model(childpi_state), "Q")
                cf = self.pi_loss.compute_cashflow_pi(
                    parent_states[:, idx.X:idx.X + 1],
                    parent_states[:, idx.Z:idx.Z + 1],
                    b_parent,
                    parent_states[:, idx.I:idx.I + 1],
                    q_parent,
                    q_pi,
                    eta,
                ).unsqueeze(1)
                for mode in m_modes:
                    result[eq][mode] = (
                        pi_exp - cf - float(Config.G) * m_by_mode[mode] * outputs[eq]["P"]
                    ).squeeze(-1)
            elif eq == "q":
                x_child = child.x_next
                z_child = child.z_next
                bar_z = outputs[eq]["bar_z"]
                qsp = outputs[eq]["Q"]
                for mode in m_modes:
                    residuals = self.q_loss.compute_main_residual(
                        q_parent,
                        b_parent,
                        bar_i,
                        [m_by_mode[mode][:, j, :] for j in range(n_child)],
                        [qsp[:, j, :] for j in range(n_child)],
                        [bar_z[:, j, :] for j in range(n_child)],
                        [x_child[:, j, :] for j in range(n_child)],
                        [z_child[:, j, :] for j in range(n_child)],
                    )
                    result[eq][mode] = torch.cat([r.reshape(bsz, 1) for r in residuals], dim=1)
            else:
                raise ValueError(f"unsupported equation: {eq}")
        return result


def _validate_fixed_state(fixed_state: Optional[Dict[str, float]]) -> Dict[str, float]:
    if fixed_state is None:
        raise ValueError("fixed_state is required for state_mode='fixed_slice'")
    missing = [key for key in REQUIRED_FIXED_STATE if key not in fixed_state]
    if missing:
        raise ValueError(f"fixed_state is missing required fields: {missing}")
    return {key: float(fixed_state[key]) for key in REQUIRED_FIXED_STATE}


def _contexts_from_fixed_state(
    b_grid: torch.Tensor,
    z_grid: torch.Tensor,
    fixed_state: Dict[str, float],
    device: torch.device,
) -> Tuple[torch.Tensor, MacroTransitionContext, Dict[str, Any]]:
    bb, zz = torch.meshgrid(b_grid.to(device), z_grid.to(device), indexing="xy")
    n = bb.numel()
    states = torch.stack([
        bb.reshape(-1),
        zz.reshape(-1),
        torch.full((n,), fixed_state["eta"], device=device),
        torch.full((n,), fixed_state["i"], device=device),
        torch.full((n,), fixed_state["x"], device=device),
        torch.full((n,), fixed_state["hatcf"], device=device),
        torch.full((n,), fixed_state["lnkf"], device=device),
    ], dim=1)
    macro = MacroTransitionContext(
        hatc_cal=torch.full((n, 1), fixed_state["hatc_cal"], device=device),
        lnk_cal=torch.full((n, 1), fixed_state["lnk_cal"], device=device),
    )
    return states, macro, {"source_indices": list(range(n)), "grid_points": int(n)}


def _load_reference_dataframe(reference_data: Any) -> pd.DataFrame:
    if isinstance(reference_data, pd.DataFrame):
        return reference_data.copy()
    path = Path(reference_data)
    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    if isinstance(payload, dict):
        return pd.DataFrame({k: v.detach().cpu().reshape(-1).numpy() if torch.is_tensor(v) else v for k, v in payload.items()})
    if torch.is_tensor(payload):
        if payload.shape[-1] < 9:
            raise ValueError("reference tensor must include seven firm columns plus hatc_cal and lnk_cal")
        cols = ["b", "z", "eta", "i", "x", "hatcf", "lnkf", "hatc_cal", "lnk_cal"]
        return pd.DataFrame(payload[:, :9].detach().cpu().numpy(), columns=cols)
    raise ValueError(f"unsupported reference_data format: {type(payload)!r}")


def _contexts_from_reference_distribution(
    b_grid: torch.Tensor,
    z_grid: torch.Tensor,
    reference_data: Any,
    *,
    n_reference_states: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, MacroTransitionContext, Dict[str, Any], Dict[str, Any]]:
    df = _load_reference_dataframe(reference_data)
    required = ["eta", "i", "x", "hatcf", "lnkf", "hatc_cal", "lnk_cal"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"reference_data is missing required columns: {missing}")
    rng = np.random.default_rng(int(seed))
    take = min(int(n_reference_states), len(df))
    indices = rng.choice(len(df), size=take, replace=False if take <= len(df) else True)
    ref = df.iloc[indices].reset_index(drop=True)
    bb, zz = torch.meshgrid(b_grid.to(device), z_grid.to(device), indexing="xy")
    states = []
    hatc = []
    lnk = []
    for _, row in ref.iterrows():
        n = bb.numel()
        states.append(torch.stack([
            bb.reshape(-1),
            zz.reshape(-1),
            torch.full((n,), float(row["eta"]), device=device),
            torch.full((n,), float(row["i"]), device=device),
            torch.full((n,), float(row["x"]), device=device),
            torch.full((n,), float(row["hatcf"]), device=device),
            torch.full((n,), float(row["lnkf"]), device=device),
        ], dim=1))
        hatc.append(torch.full((n, 1), float(row["hatc_cal"]), device=device))
        lnk.append(torch.full((n, 1), float(row["lnk_cal"]), device=device))
    parent_states = torch.cat(states, dim=0)
    macro = MacroTransitionContext(hatc_cal=torch.cat(hatc, dim=0), lnk_cal=torch.cat(lnk, dim=0))
    support = _support_metadata(df, b_grid, z_grid)
    return parent_states, macro, {"source_indices": indices.tolist(), "n_reference": int(take)}, support


def _support_metadata(df: pd.DataFrame, b_grid: torch.Tensor, z_grid: torch.Tensor) -> Dict[str, Any]:
    if not {"b", "z"}.issubset(df.columns):
        return {"support_available": False}
    b = df["b"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    bstd = float(np.std(b)) or 1.0
    zstd = float(np.std(z)) or 1.0
    points = np.stack(np.meshgrid(b_grid.cpu().numpy(), z_grid.cpu().numpy(), indexing="xy"), axis=-1).reshape(-1, 2)
    ref = np.stack([b, z], axis=1)
    d = ((points[:, None, 0] - ref[None, :, 0]) / bstd) ** 2 + ((points[:, None, 1] - ref[None, :, 1]) / zstd) ** 2
    dist = np.sqrt(d.min(axis=1))
    threshold = float(np.quantile(dist, 0.9))
    return {
        "support_available": True,
        "definition": "standardized_nearest_neighbor_distance_leq_grid_p90",
        "threshold": threshold,
        "distance": dist.tolist(),
        "in_support": (dist <= threshold).tolist(),
    }


def _aggregate_values(values: torch.Tensor, state_mode: str, n_grid: int, n_reference: int) -> Dict[str, torch.Tensor]:
    if state_mode == "fixed_slice":
        return {"value": values}
    arr = values.reshape(n_reference, n_grid)
    return {
        "mean": torch.nanmean(arr, dim=0),
        "p50": torch.nanquantile(arr, 0.50, dim=0),
        "p90": torch.nanquantile(arr, 0.90, dim=0),
        "p99": torch.nanquantile(arr, 0.99, dim=0),
        "max": torch.nan_to_num(arr, nan=-math.inf).max(dim=0).values,
    }


def evaluate_checkpoint_convergence_surfaces(
    checkpoint_paths: Sequence[str | Path],
    *,
    b_grid: Sequence[float],
    z_grid: Sequence[float],
    state_mode: str,
    sdf_checkpoint: Optional[str | Path] = None,
    hyperparams_json: Optional[str | Path] = None,
    allow_default_hyperparams: bool = False,
    fixed_state: Optional[Dict[str, float]] = None,
    reference_data: Any = None,
    n_reference_states: int = 128,
    n_child_shocks: int = 64,
    seed: int = 2026,
    equations: Sequence[str] = ("p0", "pi", "q"),
    m_modes: Sequence[str] = ("train", "raw"),
    branch_weights: Optional[torch.Tensor] = None,
    parent_chunk_size: int = 512,
    child_chunk_size: int = 8192,
    device: Optional[str | torch.device] = None,
    output_dir: Optional[str | Path] = None,
) -> ConvergenceSurfaceResult:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    b_grid_t = torch.tensor(list(b_grid), dtype=torch.float32, device=device)
    z_grid_t = torch.tensor(list(z_grid), dtype=torch.float32, device=device)
    n_grid = int(b_grid_t.numel() * z_grid_t.numel())
    if state_mode == "fixed_slice":
        fixed = _validate_fixed_state(fixed_state)
        parent_states, macro, context_meta = _contexts_from_fixed_state(b_grid_t, z_grid_t, fixed, device)
        support = {"support_available": False}
        n_reference = 1
    elif state_mode == "reference_distribution":
        if reference_data is None:
            raise ValueError("reference_data is required for reference_distribution")
        parent_states, macro, context_meta, support = _contexts_from_reference_distribution(
            b_grid_t,
            z_grid_t,
            reference_data,
            n_reference_states=n_reference_states,
            seed=seed,
            device=device,
        )
        n_reference = int(context_meta["n_reference"])
    else:
        raise ValueError(f"unsupported state_mode: {state_mode}")

    shock_bank = ConvergenceShockBank.create(
        parent_states.shape[0],
        n_child_shocks,
        seed=seed,
        device=device,
        dtype=parent_states.dtype,
    )
    rows: List[Dict[str, Any]] = []
    raw_tensors: Dict[str, Any] = {}
    checkpoint_metadata = []

    loaded = [
        load_analysis_checkpoint(
            path,
            sdf_checkpoint=sdf_checkpoint if len(checkpoint_paths) == 1 else None,
            hyperparams_json=hyperparams_json if len(checkpoint_paths) == 1 else None,
            allow_default_hyperparams=allow_default_hyperparams,
            device=device,
        )
        for path in checkpoint_paths
    ]
    models_for_state = [m for ckpt in loaded for m in (ckpt.models["policy_value"], ckpt.models["sdf_fc1"])]
    with preserve_analysis_state(models_for_state):
        for ckpt_idx, ckpt in enumerate(loaded):
            metadata = dict(ckpt.metadata)
            checkpoint_metadata.append(metadata)
            backend = VectorizedBellmanSurfaceBackend(ckpt.models["policy_value"], ckpt.hyperparams)
            per_metric_values: Dict[Tuple[str, str, str], List[torch.Tensor]] = {}
            for start in range(0, parent_states.shape[0], int(parent_chunk_size)):
                end = min(start + int(parent_chunk_size), parent_states.shape[0])
                ps = parent_states[start:end]
                macro_chunk = MacroTransitionContext(
                    hatc_cal=macro.hatc_cal[start:end],
                    lnk_cal=macro.lnk_cal[start:end],
                )
                sb = ConvergenceShockBank(
                    eps_x=shock_bank.eps_x[start:end],
                    eps_z=shock_bank.eps_z[start:end],
                    u_eta=shock_bank.u_eta[start:end],
                    u_i=shock_bank.u_i[start:end],
                    seed=shock_bank.seed,
                )
                child = build_child_exogenous_bundle(
                    ckpt.models["sdf_fc1"],
                    ps,
                    macro_chunk,
                    sb,
                    branch_weights=branch_weights,
                )
                residuals = backend.compute_signed_residuals(
                    ps,
                    child,
                    equations=equations,
                    m_modes=m_modes,
                    child_chunk_size=child_chunk_size,
                )
                for eq in equations:
                    for mode in m_modes:
                        reduced = reduce_signed_surface(residuals[eq][mode], child.branch_weights)
                        for metric, tensor in reduced.items():
                            per_metric_values.setdefault((eq, mode, metric), []).append(tensor.detach().cpu())

            checkpoint_label = f"checkpoint_{ckpt_idx}"
            raw_tensors[checkpoint_label] = {}
            bb, zz = torch.meshgrid(b_grid_t.cpu(), z_grid_t.cpu(), indexing="xy")
            flat_b = bb.reshape(-1).numpy()
            flat_z = zz.reshape(-1).numpy()
            support_distance = support.get("distance")
            in_support = support.get("in_support")
            for key, chunks in per_metric_values.items():
                eq, mode, metric = key
                values = torch.cat(chunks, dim=0)
                raw_tensors[checkpoint_label][f"{eq}.{mode}.{metric}"] = values
                aggs = _aggregate_values(values, state_mode, n_grid, n_reference)
                finite = torch.isfinite(values.reshape(n_reference, n_grid) if state_mode != "fixed_slice" else values.reshape(1, n_grid))
                finite_ratio = finite.float().mean(dim=0).numpy()
                n_finite = finite.sum(dim=0).numpy()
                for agg_name, agg_vals in aggs.items():
                    agg_np = agg_vals.reshape(-1).numpy()
                    for pos, val in enumerate(agg_np):
                        grid_pos = pos % n_grid
                        rows.append({
                            "checkpoint": checkpoint_label,
                            "checkpoint_hash": metadata.get("checkpoint_sha256"),
                            "policy_state_hash": metadata.get("policy_state_hash"),
                            "sdf_state_hash": metadata.get("sdf_state_hash"),
                            "equation": eq,
                            "m_mode": mode,
                            "metric": metric,
                            "aggregation": agg_name,
                            "b": float(flat_b[grid_pos]),
                            "z": float(flat_z[grid_pos]),
                            "value": float(val),
                            "finite_ratio": float(finite_ratio[grid_pos]),
                            "n_reference_finite": int(n_finite[grid_pos]),
                            "n_reference_states": int(n_reference),
                            "n_child_shocks": int(n_child_shocks),
                            "seed": int(seed),
                            "state_mode": state_mode,
                            "in_support": None if in_support is None else bool(in_support[grid_pos]),
                            "support_distance": None if support_distance is None else float(support_distance[grid_pos]),
                        })

    table = pd.DataFrame(rows)
    manifests = {
        "surface_family": SURFACE_FAMILY,
        "state_mode": state_mode,
        "b_grid": [float(x) for x in b_grid],
        "z_grid": [float(x) for x in z_grid],
        "m_parameters": [
            {
                "pv_use_clipped_m": bool(getattr(ckpt.hyperparams, "pv_use_clipped_m", True)),
                "pv_m_clamp_min": float(getattr(ckpt.hyperparams, "pv_m_clamp_min", 0.7)),
                "pv_m_clamp_max": float(getattr(ckpt.hyperparams, "pv_m_clamp_max", 1.3)),
                "q_use_detached_m": bool(getattr(ckpt.hyperparams, "q_use_detached_m", True)),
                "q_m_clamp_min": float(getattr(ckpt.hyperparams, "q_m_clamp_min", 0.5)),
                "q_m_clamp_max": float(getattr(ckpt.hyperparams, "q_m_clamp_max", 1.5)),
            }
            for ckpt in loaded
        ],
        "context_metadata": context_meta,
    }
    result = ConvergenceSurfaceResult(
        long_table=table,
        raw_tensors=raw_tensors,
        manifests=manifests,
        checkpoint_metadata=checkpoint_metadata,
        shock_bank_metadata={"seed": int(seed), "n_child_shocks": int(n_child_shocks)},
        support_metadata=support,
        disabled_metrics={"policy_regret": "disabled_first_version"},
    )
    if output_dir is not None:
        save_convergence_surface_result(result, output_dir)
    return result


def save_convergence_surface_result(result: ConvergenceSurfaceResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.long_table.to_csv(output / "surface_long.csv", index=False)
    try:
        result.long_table.to_parquet(output / "surface_long.parquet", index=False)
    except Exception:
        pass
    torch.save(result.raw_tensors, output / "raw_surface.pt")
    manifest = {
        **result.manifests,
        "checkpoint_metadata": result.checkpoint_metadata,
        "shock_bank_metadata": result.shock_bank_metadata,
        "support_metadata": result.support_metadata,
        "disabled_metrics": result.disabled_metrics,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_primary_pngs(result.long_table, output)


def _write_primary_pngs(table: pd.DataFrame, output: Path, *, include_raw: bool = False) -> None:
    import matplotlib.pyplot as plt

    if table.empty:
        return
    selected = [
        ("conditional_abs", "train", "mean"),
        ("conditional_abs", "train", "value"),
        ("conditional_abs", "train", "p90"),
        ("realized_abs", "train", "mean"),
        ("realized_abs", "train", "value"),
        ("child_dispersion", "train", "mean"),
        ("child_dispersion", "train", "value"),
        ("mc_standard_error", "train", "mean"),
        ("mc_standard_error", "train", "value"),
    ]
    if include_raw:
        selected.extend([(m, "raw", a) for m, _, a in selected])
    for metric, mode, agg in selected:
        subset_all = table[(table["metric"] == metric) & (table["m_mode"] == mode) & (table["aggregation"] == agg)]
        if subset_all.empty:
            continue
        vmin = subset_all["value"].replace([np.inf, -np.inf], np.nan).min()
        vmax = subset_all["value"].replace([np.inf, -np.inf], np.nan).max()
        metric_name = "conditional" if metric == "conditional_abs" else metric
        for (checkpoint, equation), sub in subset_all.groupby(["checkpoint", "equation"]):
            pivot = sub.pivot_table(index="z", columns="b", values="value", aggfunc="mean").sort_index()
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(
                pivot.values,
                origin="lower",
                aspect="auto",
                extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("b")
            ax.set_ylabel("z")
            ax.set_title(f"{checkpoint} {equation} {metric} {mode} {agg}")
            fig.colorbar(im, ax=ax, label=metric)
            fig.tight_layout()
            fig.savefig(output / f"{equation}_{metric_name}_{mode}_{agg}.png")
            plt.close(fig)
        checkpoints = list(subset_all["checkpoint"].drop_duplicates())
        if len(checkpoints) >= 2:
            base = checkpoints[0]
            for other in checkpoints[1:]:
                for equation in subset_all["equation"].drop_duplicates():
                    a = subset_all[(subset_all["checkpoint"] == base) & (subset_all["equation"] == equation)]
                    b = subset_all[(subset_all["checkpoint"] == other) & (subset_all["equation"] == equation)]
                    if a.empty or b.empty:
                        continue
                    pa = a.pivot_table(index="z", columns="b", values="value", aggfunc="mean").sort_index()
                    pb = b.pivot_table(index="z", columns="b", values="value", aggfunc="mean").sort_index()
                    delta = pb - pa
                    fig, ax = plt.subplots(figsize=(5, 4))
                    im = ax.imshow(
                        delta.values,
                        origin="lower",
                        aspect="auto",
                        extent=[delta.columns.min(), delta.columns.max(), delta.index.min(), delta.index.max()],
                    )
                    ax.set_xlabel("b")
                    ax.set_ylabel("z")
                    ax.set_title(f"delta {other} minus {base} {equation} {metric}")
                    fig.colorbar(im, ax=ax, label=f"delta {metric}")
                    fig.tight_layout()
                    fig.savefig(output / f"delta_{other}_minus_{base}_{equation}_{metric_name}_{mode}_{agg}.png")
                    plt.close(fig)
