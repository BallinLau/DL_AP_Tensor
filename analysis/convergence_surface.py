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

from config import HyperParams
from losses import P0Loss, PILoss, QLoss
from utils.firm_transition import apply_refinancing_policy

from .checkpoint_loader import (
    AnalysisCheckpoint,
    CheckpointSpec,
    load_analysis_checkpoint,
    load_analysis_checkpoint_spec,
)
from .convergence_transition import (
    ChildExogenousBundle,
    ConvergenceShockBank,
    FirmStateIndex,
    MacroTransitionContext,
    build_child_exogenous_bundle,
)
from .economic_config import AnalysisEconomicConfig


SURFACE_FAMILY = "bellman_fixed_point_conditional_mean_v1"
REQUIRED_FIXED_STATE = ("eta", "i", "x", "hatcf", "lnkf", "hatc_cal", "lnk_cal")
FIXED_GRID_MODE = "fixed_grid"


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
def preserve_global_rng():
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@contextmanager
def preserve_analysis_state(models: Iterable[torch.nn.Module]):
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


def _policy_get(out: Any, name: str) -> torch.Tensor:
    if isinstance(out, dict):
        return out[name]
    return getattr(out, name)


def _tensor_hash(tensors: Sequence[torch.Tensor]) -> str:
    import hashlib

    h = hashlib.sha256()
    for tensor in tensors:
        arr = tensor.detach().cpu().contiguous()
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.numpy().tobytes())
    return h.hexdigest()


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
        economic_config: Optional[AnalysisEconomicConfig] = None,
        *,
        p0_loss: Optional[P0Loss] = None,
        pi_loss: Optional[PILoss] = None,
        q_loss: Optional[QLoss] = None,
    ) -> None:
        self.policy_model = policy_model
        self.hyperparams = hyperparams
        self.economic_config = economic_config or AnalysisEconomicConfig.from_current_config()
        cfg = self.economic_config
        self.p0_loss = p0_loss or P0Loss(
            delta=cfg.DELTA,
            tau=cfg.TAU,
            kappa_b=cfg.KAPPA_B,
            kappa_e=cfg.KAPPA_E,
            aio_weight=cfg.AIO_WEIGHT,
            alpha_z=cfg.ALPHA_Z,
            beta_z=cfg.BETA_Z,
            z0=cfg.Z0,
        )
        self.pi_loss = pi_loss or PILoss(
            delta=cfg.DELTA,
            tau=cfg.TAU,
            g=cfg.G,
            kappa_b=cfg.KAPPA_B,
            kappa_e=cfg.KAPPA_E,
            aio_weight=cfg.AIO_WEIGHT,
            alpha_z=cfg.ALPHA_Z,
            beta_z=cfg.BETA_Z,
            z0=cfg.Z0,
        )
        self.q_loss = q_loss or QLoss(
            delta=cfg.DELTA,
            phi=cfg.PHI,
            g=cfg.G,
            aio_weight=cfg.AIO_WEIGHT,
            alpha_z=cfg.ALPHA_Z,
            beta_z=cfg.BETA_Z,
            z0=cfg.Z0,
        )

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
        multiplier = bar_i * (float(self.economic_config.G) - 1.0) + 1.0
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
            # The diagnostic surface must be exactly invariant to requested
            # chunk sizes. Use a canonical full child batch after the workload
            # guard has accepted the run; the argument remains part of the API
            # for future streaming backends.
            step = flat.shape[0]
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
                        pi_exp - cf - float(self.economic_config.G) * m_by_mode[mode] * outputs[eq]["P"]
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
) -> Tuple[torch.Tensor, MacroTransitionContext, torch.Tensor, Dict[str, Any]]:
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
    reference_index = torch.zeros(n, dtype=torch.long, device=device)
    return states, macro, reference_index, {"source_indices": [0], "grid_points": int(n)}


def _load_reference_dataframe(reference_data: Any) -> pd.DataFrame:
    if isinstance(reference_data, pd.DataFrame):
        return _normalize_reference_columns(reference_data.copy())
    if isinstance(reference_data, dict):
        if "firm" in reference_data and "macro" in reference_data:
            return _join_firm_macro_reference(reference_data["firm"], reference_data["macro"])
        if "parent" in reference_data:
            parent = reference_data["parent"]
            if torch.is_tensor(parent):
                if parent.shape[-1] < 7:
                    raise ValueError("reference parent tensor must have at least seven firm columns")
                cols = ["b", "z", "eta", "i", "x", "hatcf", "lnkf"]
                df = pd.DataFrame(parent[:, :7].detach().cpu().numpy(), columns=cols)
            else:
                df = pd.DataFrame(parent)
            if "hatc_cal" not in reference_data or "lnk_cal" not in reference_data:
                raise ValueError("tensor/batch reference bundle must include hatc_cal and lnk_cal")
            df["hatc_cal"] = torch.as_tensor(reference_data["hatc_cal"]).detach().cpu().reshape(-1).numpy()
            df["lnk_cal"] = torch.as_tensor(reference_data["lnk_cal"]).detach().cpu().reshape(-1).numpy()
            return _normalize_reference_columns(df)
        return _normalize_reference_columns(pd.DataFrame({
            k: v.detach().cpu().reshape(-1).numpy() if torch.is_tensor(v) else v
            for k, v in reference_data.items()
        }))
    if isinstance(reference_data, list):
        frames = []
        for batch in reference_data:
            frames.append(_load_reference_dataframe(batch))
        if not frames:
            raise ValueError("reference batch list is empty")
        return _normalize_reference_columns(pd.concat(frames, ignore_index=True))
    path = Path(reference_data)
    if path.suffix in {".pkl", ".pickle"}:
        return _attach_sibling_macro_if_available(_normalize_reference_columns(pd.read_pickle(path)), path)
    if path.suffix == ".csv":
        return _attach_sibling_macro_if_available(_normalize_reference_columns(pd.read_csv(path)), path)
    payload = torch.load(path, map_location="cpu")
    return _load_reference_dataframe(payload)
    raise ValueError(f"unsupported reference_data format: {type(payload)!r}")


def _normalize_reference_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "B": "b",
        "Z": "z",
        "ETA": "eta",
        "I": "i",
        "X": "x",
        "Hatcf": "hatcf",
        "HATCF": "hatcf",
        "LnKF": "lnkf",
        "LNKF": "lnkf",
        "Hatc_cal": "hatc_cal",
        "HATC_CAL": "hatc_cal",
        "Hatc": "hatc_cal",
        "HATC": "hatc_cal",
        "LnK_cal": "lnk_cal",
        "LNK_CAL": "lnk_cal",
        "LnK": "lnk_cal",
        "LNK": "lnk_cal",
    }
    renamed = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
    return _coalesce_duplicate_columns(renamed)


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.has_duplicates:
        return df
    out = pd.DataFrame(index=df.index)
    for col in dict.fromkeys(df.columns):
        subset = df.loc[:, df.columns == col]
        if subset.shape[1] == 1:
            out[col] = subset.iloc[:, 0]
        else:
            out[col] = subset.bfill(axis=1).iloc[:, 0]
    return out


def _row_float(row: pd.Series, key: str) -> float:
    value = row[key]
    if isinstance(value, pd.Series):
        non_null = value.dropna()
        if non_null.empty:
            raise ValueError(f"reference row field {key!r} is all NaN")
        value = non_null.iloc[0]
    return float(value)


def _attach_sibling_macro_if_available(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = _normalize_reference_columns(df)
    if {"hatc_cal", "lnk_cal"}.issubset(df.columns):
        return df
    candidates = [
        path.with_name(path.stem + "_macro" + path.suffix),
        path.with_name(path.name.replace(".pkl", "_macro.pkl")),
    ]
    if "_macro" not in path.stem:
        stem = path.stem
        if stem.startswith("ep") and "_stage_" in stem:
            parts = stem.split("_stage_", 1)
            candidates.append(path.with_name(f"{parts[0]}_stage_{parts[1]}_macro{path.suffix}"))
    for macro_path in dict.fromkeys(candidates):
        if not macro_path.exists() or macro_path == path:
            continue
        macro_df = _normalize_reference_columns(pd.read_pickle(macro_path) if macro_path.suffix in {".pkl", ".pickle"} else pd.read_csv(macro_path))
        joined = _join_firm_macro_reference(df, macro_df)
        if {"hatc_cal", "lnk_cal"}.issubset(joined.columns):
            joined.attrs["macro_source_path"] = str(macro_path)
            return joined
    return df


def _join_firm_macro_reference(firm: Any, macro: Any) -> pd.DataFrame:
    firm_df = _normalize_reference_columns(firm.copy() if isinstance(firm, pd.DataFrame) else pd.DataFrame(firm))
    macro_df = _normalize_reference_columns(macro.copy() if isinstance(macro, pd.DataFrame) else pd.DataFrame(macro))
    key_candidates = [
        [key for key in ("path", "t", "branch", "ID") if key in firm_df.columns and key in macro_df.columns],
        [key for key in ("path", "t", "branch") if key in firm_df.columns and key in macro_df.columns],
        [key for key in ("path", "t") if key in firm_df.columns and key in macro_df.columns],
    ]
    key_candidates = [keys for keys in key_candidates if keys]
    if not key_candidates:
        if len(macro_df) == len(firm_df):
            out = firm_df.copy()
            for col in ("hatc_cal", "lnk_cal"):
                if col in macro_df.columns:
                    out[col] = macro_df[col].to_numpy()
            return _normalize_reference_columns(out)
        raise ValueError("firm/macro reference bundle requires join keys or matching row counts")
    cal_cols = [c for c in ("hatc_cal", "lnk_cal") if c in macro_df.columns]
    best = None
    best_nonmissing = -1
    for keys in key_candidates:
        macro_cols = keys + cal_cols
        joined = firm_df.merge(macro_df[macro_cols].drop_duplicates(keys), on=keys, how="left")
        nonmissing = int(joined[cal_cols].notna().all(axis=1).sum()) if cal_cols else 0
        if nonmissing > best_nonmissing:
            best = joined
            best_nonmissing = nonmissing
        if cal_cols and nonmissing == len(joined):
            break
    if best is None:
        raise ValueError("could not join firm and macro reference data")
    return _normalize_reference_columns(best)


def _select_parent_reference_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "t" in df.columns:
        non_null_t = df["t"].dropna()
        first_t = non_null_t.iloc[0] if len(non_null_t) else None
        if pd.api.types.is_string_dtype(df["t"]) or isinstance(first_t, str):
            parent = df[df["t"] == "t"].copy()
            if not parent.empty:
                return parent
    if "branch" in df.columns:
        branch_num = pd.to_numeric(df["branch"], errors="coerce")
        parent = df[branch_num == -1].copy()
        if not parent.empty:
            return parent
    return df


def _contexts_from_reference_distribution(
    b_grid: torch.Tensor,
    z_grid: torch.Tensor,
    reference_data: Any,
    *,
    n_reference_states: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, MacroTransitionContext, torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    df = _load_reference_dataframe(reference_data)
    if df.empty:
        raise ValueError("reference dataframe is empty")
    df = _select_parent_reference_rows(df)
    if df.empty:
        raise ValueError("reference dataframe has no usable parent rows")
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
            torch.full((n,), _row_float(row, "eta"), device=device),
            torch.full((n,), _row_float(row, "i"), device=device),
            torch.full((n,), _row_float(row, "x"), device=device),
            torch.full((n,), _row_float(row, "hatcf"), device=device),
            torch.full((n,), _row_float(row, "lnkf"), device=device),
        ], dim=1))
        hatc.append(torch.full((n, 1), _row_float(row, "hatc_cal"), device=device))
        lnk.append(torch.full((n, 1), _row_float(row, "lnk_cal"), device=device))
    parent_states = torch.cat(states, dim=0)
    macro = MacroTransitionContext(hatc_cal=torch.cat(hatc, dim=0), lnk_cal=torch.cat(lnk, dim=0))
    reference_index = torch.arange(take, device=device, dtype=torch.long).repeat_interleave(n)
    support = _support_metadata(ref, b_grid, z_grid)
    return parent_states, macro, reference_index, {
        "source_indices": indices.tolist(),
        "n_reference": int(take),
        "reference_adapter_type": type(reference_data).__name__,
    }, support


def _support_metadata(
    df: pd.DataFrame,
    b_grid: torch.Tensor,
    z_grid: torch.Tensor,
    *,
    loo_quantile: float = 0.95,
    support_radius: Optional[float] = None,
) -> Dict[str, Any]:
    if not {"b", "z"}.issubset(df.columns):
        return {"support_available": False}
    ref_df = df[["b", "z"]].replace([np.inf, -np.inf], np.nan).dropna()
    if ref_df.empty:
        raise ValueError("reference support data has no finite b,z rows")
    b = ref_df["b"].to_numpy(dtype=float)
    z = ref_df["z"].to_numpy(dtype=float)
    bstd = float(np.std(b))
    zstd = float(np.std(z))
    if not np.isfinite(bstd) or bstd == 0.0:
        bstd = 1.0
    if not np.isfinite(zstd) or zstd == 0.0:
        zstd = 1.0
    points = np.stack(np.meshgrid(b_grid.cpu().numpy(), z_grid.cpu().numpy(), indexing="xy"), axis=-1).reshape(-1, 2)
    ref = np.stack([b, z], axis=1)
    scaled_ref = np.column_stack([ref[:, 0] / bstd, ref[:, 1] / zstd])
    scaled_grid = np.column_stack([points[:, 0] / bstd, points[:, 1] / zstd])
    if support_radius is None:
        if len(scaled_ref) < 2:
            return {
                "support_available": False,
                "definition": "standardized_reference_loo_nearest_neighbor",
                "reference_count": int(len(scaled_ref)),
                "reason": "fewer_than_two_reference_points",
            }
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(scaled_ref)
            loo = tree.query(scaled_ref, k=2)[0][:, 1]
            dist = tree.query(scaled_grid, k=1)[0]
        except Exception:
            diff = scaled_ref[:, None, :] - scaled_ref[None, :, :]
            full = np.sqrt((diff * diff).sum(axis=-1))
            np.fill_diagonal(full, np.inf)
            loo = full.min(axis=1)
            chunks = []
            for start in range(0, len(scaled_grid), 4096):
                g = scaled_grid[start:start + 4096]
                d = np.sqrt(((g[:, None, :] - scaled_ref[None, :, :]) ** 2).sum(axis=-1)).min(axis=1)
                chunks.append(d)
            dist = np.concatenate(chunks)
        threshold = float(np.quantile(loo, float(loo_quantile)))
    else:
        threshold = float(support_radius)
        try:
            from scipy.spatial import cKDTree

            dist = cKDTree(scaled_ref).query(scaled_grid, k=1)[0]
        except Exception:
            chunks = []
            for start in range(0, len(scaled_grid), 4096):
                g = scaled_grid[start:start + 4096]
                chunks.append(np.sqrt(((g[:, None, :] - scaled_ref[None, :, :]) ** 2).sum(axis=-1)).min(axis=1))
            dist = np.concatenate(chunks)
    return {
        "support_available": True,
        "definition": "standardized_reference_loo_nearest_neighbor",
        "standardization": {"b_std": bstd, "z_std": zstd},
        "reference_count": int(len(scaled_ref)),
        "loo_quantile": None if support_radius is not None else float(loo_quantile),
        "threshold": threshold,
        "distance": dist.tolist(),
        "in_support": (dist <= threshold).tolist(),
    }


def _aggregate_values(values: torch.Tensor, state_mode: str, n_grid: int, n_reference: int) -> Dict[str, torch.Tensor]:
    if state_mode == "fixed_slice":
        return {"value": values}
    arr = values.reshape(n_reference, n_grid)
    finite = torch.isfinite(arr)
    any_finite = finite.any(dim=0)
    nan = torch.full((n_grid,), float("nan"), dtype=arr.dtype)
    arr_for_quantile = arr.clone()
    arr_for_quantile[~finite] = float("nan")
    max_vals = torch.where(any_finite, torch.nan_to_num(arr, nan=-math.inf).max(dim=0).values, nan)
    return {
        "mean": torch.where(any_finite, torch.nanmean(arr_for_quantile, dim=0), nan),
        "p50": torch.where(any_finite, torch.nanquantile(arr_for_quantile, 0.50, dim=0), nan),
        "p90": torch.where(any_finite, torch.nanquantile(arr_for_quantile, 0.90, dim=0), nan),
        "p99": torch.where(any_finite, torch.nanquantile(arr_for_quantile, 0.99, dim=0), nan),
        "max": max_vals,
    }


def _as_checkpoint_specs(
    checkpoint_paths: Sequence[str | Path | CheckpointSpec],
    *,
    sdf_checkpoint: Optional[str | Path],
    hyperparams_json: Optional[str | Path],
    config_json: Optional[str | Path],
    checkpoint_labels: Optional[Sequence[str]],
) -> List[CheckpointSpec]:
    specs: List[CheckpointSpec] = []
    for i, item in enumerate(checkpoint_paths):
        if isinstance(item, CheckpointSpec):
            spec = item
        else:
            spec = CheckpointSpec(
                checkpoint_path=item,
                sdf_checkpoint=sdf_checkpoint if len(checkpoint_paths) == 1 else None,
                hyperparams_json=hyperparams_json if len(checkpoint_paths) == 1 else None,
                config_json=config_json if len(checkpoint_paths) == 1 else None,
            )
        if checkpoint_labels is not None:
            if i >= len(checkpoint_labels):
                raise ValueError("checkpoint_labels length must match checkpoint list length")
            spec = CheckpointSpec(
                checkpoint_path=spec.checkpoint_path,
                policy_checkpoint=spec.policy_checkpoint,
                sdf_checkpoint=spec.sdf_checkpoint,
                hyperparams_json=spec.hyperparams_json,
                config_json=spec.config_json,
                label=checkpoint_labels[i],
            )
        specs.append(spec)
    labels = [s.label for s in specs if s.label is not None]
    if len(labels) != len(set(labels)):
        raise ValueError("checkpoint labels must be unique")
    if len(specs) > 1:
        for spec in specs:
            raw_like = spec.policy_checkpoint is not None or (
                spec.checkpoint_path is not None and Path(spec.checkpoint_path).suffix in {".pt", ".pth"}
            )
            if spec.policy_checkpoint is not None and (spec.sdf_checkpoint is None or spec.hyperparams_json is None):
                raise ValueError("multiple raw checkpoint specs must provide their own sdf and hyperparams paths")
    return specs


def _validate_surface_inputs(
    checkpoint_paths: Sequence[Any],
    b_grid: Sequence[float],
    z_grid: Sequence[float],
    n_child_shocks: int,
    n_reference_states: int,
    parent_chunk_size: int,
    child_chunk_size: int,
    equations: Sequence[str],
    m_modes: Sequence[str],
) -> None:
    if not checkpoint_paths:
        raise ValueError("checkpoint list must be non-empty")
    b_vals = np.asarray(list(b_grid), dtype=float)
    z_vals = np.asarray(list(z_grid), dtype=float)
    if b_vals.size == 0 or not np.isfinite(b_vals).all():
        raise ValueError("b_grid must be non-empty and finite")
    if z_vals.size == 0 or not np.isfinite(z_vals).all():
        raise ValueError("z_grid must be non-empty and finite")
    if int(n_child_shocks) < 2:
        raise ValueError("n_child_shocks must be >= 2")
    if int(n_reference_states) < 1:
        raise ValueError("n_reference_states must be >= 1")
    if int(parent_chunk_size) < 1:
        raise ValueError("parent_chunk_size must be >= 1")
    if int(child_chunk_size) < 1:
        raise ValueError("child_chunk_size must be >= 1")
    bad_eq = set(equations) - {"p0", "pi", "q"}
    if bad_eq:
        raise ValueError(f"unsupported equations: {sorted(bad_eq)}")
    bad_modes = set(m_modes) - {"train", "raw"}
    if bad_modes:
        raise ValueError(f"unsupported m_modes: {sorted(bad_modes)}")


def _fixed_boundary_status(phat: np.ndarray) -> str:
    finite = phat[np.isfinite(phat)]
    if finite.size == 0:
        return "unknown_nonfinite"
    if np.all(finite > 0):
        return "all_survival"
    if np.all(finite < 0):
        return "all_default"
    return "observed"


def _extract_default_boundary_rows(
    *,
    checkpoint: str,
    b_values: np.ndarray,
    z_values: np.ndarray,
    phat_grid: np.ndarray,
) -> List[Dict[str, Any]]:
    import matplotlib.pyplot as plt

    if _fixed_boundary_status(phat_grid) != "observed":
        return []
    fig, ax = plt.subplots()
    try:
        contour = ax.contour(b_values, z_values, phat_grid, levels=[0.0])
        rows: List[Dict[str, Any]] = []
        component_id = 0
        for segs in contour.allsegs:
            for seg in segs:
                if seg.size == 0:
                    continue
                for point_index, (b, z) in enumerate(seg):
                    rows.append({
                        "checkpoint": checkpoint,
                        "component_id": int(component_id),
                        "point_index": int(point_index),
                        "b": float(b),
                        "z": float(z),
                    })
                component_id += 1
        return rows
    finally:
        plt.close(fig)


def _resolve_branch_weights(
    branch_weights: Optional[torch.Tensor],
    *,
    n_reference: int,
    n_grid: int,
    n_child: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if branch_weights is None:
        return None
    weights = torch.as_tensor(branch_weights, device=device, dtype=dtype)
    n_parent = n_reference * n_grid
    if weights.ndim == 1:
        if weights.shape[0] != n_child:
            raise ValueError(f"branch_weights length must be {n_child}, got {weights.shape[0]}")
        weights = weights.reshape(1, n_child).expand(n_parent, n_child)
    elif weights.ndim == 2:
        if tuple(weights.shape) == (n_reference, n_child):
            weights = weights.repeat_interleave(n_grid, dim=0)
        elif tuple(weights.shape) == (n_parent, n_child):
            weights = weights
        else:
            raise ValueError(
                f"branch_weights shape must be [{n_child}], {(n_reference, n_child)}, or {(n_parent, n_child)}; "
                f"got {tuple(weights.shape)}"
            )
    else:
        raise ValueError("branch_weights must have shape [J], [N_reference,J], or [N_reference*N_grid,J]")
    return weights


def evaluate_checkpoint_convergence_surfaces(
    checkpoint_paths: Sequence[str | Path | CheckpointSpec],
    *,
    b_grid: Sequence[float],
    z_grid: Sequence[float],
    state_mode: str,
    sdf_checkpoint: Optional[str | Path] = None,
    hyperparams_json: Optional[str | Path] = None,
    config_json: Optional[str | Path] = None,
    allow_default_hyperparams: bool = False,
    allow_current_config: bool = False,
    checkpoint_labels: Optional[Sequence[str]] = None,
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
    support_radius: Optional[float] = None,
    max_child_state_evals: Optional[int] = None,
    allow_large_run: bool = False,
    include_raw_plots: bool = False,
    apply_support_mask: bool = True,
    include_signed: bool = False,
    residual_threshold: Optional[float] = None,
    log_residual_scale: bool = False,
) -> ConvergenceSurfaceResult:
    with preserve_global_rng():
        return _evaluate_checkpoint_convergence_surfaces_impl(
            checkpoint_paths,
            b_grid=b_grid,
            z_grid=z_grid,
            state_mode=state_mode,
            sdf_checkpoint=sdf_checkpoint,
            hyperparams_json=hyperparams_json,
            config_json=config_json,
            allow_default_hyperparams=allow_default_hyperparams,
            allow_current_config=allow_current_config,
            checkpoint_labels=checkpoint_labels,
            fixed_state=fixed_state,
            reference_data=reference_data,
            n_reference_states=n_reference_states,
            n_child_shocks=n_child_shocks,
            seed=seed,
            equations=equations,
            m_modes=m_modes,
            branch_weights=branch_weights,
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size,
            device=device,
            output_dir=output_dir,
            support_radius=support_radius,
            max_child_state_evals=max_child_state_evals,
            allow_large_run=allow_large_run,
            include_raw_plots=include_raw_plots,
            apply_support_mask=apply_support_mask,
            include_signed=include_signed,
            residual_threshold=residual_threshold,
            log_residual_scale=log_residual_scale,
        )


def _evaluate_checkpoint_convergence_surfaces_impl(
    checkpoint_paths: Sequence[str | Path | CheckpointSpec],
    *,
    b_grid: Sequence[float],
    z_grid: Sequence[float],
    state_mode: str,
    sdf_checkpoint: Optional[str | Path],
    hyperparams_json: Optional[str | Path],
    config_json: Optional[str | Path],
    allow_default_hyperparams: bool,
    allow_current_config: bool,
    checkpoint_labels: Optional[Sequence[str]],
    fixed_state: Optional[Dict[str, float]],
    reference_data: Any,
    n_reference_states: int,
    n_child_shocks: int,
    seed: int,
    equations: Sequence[str],
    m_modes: Sequence[str],
    branch_weights: Optional[torch.Tensor],
    parent_chunk_size: int,
    child_chunk_size: int,
    device: Optional[str | torch.device],
    output_dir: Optional[str | Path],
    support_radius: Optional[float],
    max_child_state_evals: Optional[int],
    allow_large_run: bool,
    include_raw_plots: bool,
    apply_support_mask: bool,
    include_signed: bool,
    residual_threshold: Optional[float],
    log_residual_scale: bool,
) -> ConvergenceSurfaceResult:
    _validate_surface_inputs(
        checkpoint_paths,
        b_grid,
        z_grid,
        n_child_shocks,
        n_reference_states,
        parent_chunk_size,
        child_chunk_size,
        equations,
        m_modes,
    )
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    b_grid_t = torch.tensor(list(b_grid), dtype=torch.float32, device=device)
    z_grid_t = torch.tensor(list(z_grid), dtype=torch.float32, device=device)
    n_grid = int(b_grid_t.numel() * z_grid_t.numel())
    if state_mode in {"fixed_slice", FIXED_GRID_MODE}:
        fixed = _validate_fixed_state(fixed_state)
        parent_states, macro, parent_reference_index, context_meta = _contexts_from_fixed_state(b_grid_t, z_grid_t, fixed, device)
        context_meta["fixed_state"] = dict(fixed)
        support = {"support_available": False}
        n_reference = 1
    elif state_mode == "reference_distribution":
        if reference_data is None:
            raise ValueError("reference_data is required for reference_distribution")
        parent_states, macro, parent_reference_index, context_meta, support = _contexts_from_reference_distribution(
            b_grid_t,
            z_grid_t,
            reference_data,
            n_reference_states=n_reference_states,
            seed=seed,
            device=device,
        )
        if support_radius is not None:
            support = _support_metadata(_load_reference_dataframe(reference_data), b_grid_t, z_grid_t, support_radius=support_radius)
        n_reference = int(context_meta["n_reference"])
    else:
        raise ValueError(f"unsupported state_mode: {state_mode}")
    if state_mode == FIXED_GRID_MODE:
        m_modes = ("train",)

    n_parent_states = int(parent_states.shape[0])
    n_child_state_evals = n_parent_states * int(n_child_shocks) * len(tuple(equations))
    workload = {
        "n_parent_states": n_parent_states,
        "n_child_state_evals": n_child_state_evals,
        "estimated_value_head_evals": n_child_state_evals,
    }
    if max_child_state_evals is not None and n_child_state_evals > int(max_child_state_evals) and not allow_large_run:
        raise ValueError(
            f"workload guard blocked run: n_child_state_evals={n_child_state_evals} "
            f"> max_child_state_evals={max_child_state_evals}; set allow_large_run=True to proceed"
        )

    shock_bank = ConvergenceShockBank.create(
        n_reference,
        n_child_shocks,
        seed=seed,
        device=device,
        dtype=parent_states.dtype,
    )
    weights_all = _resolve_branch_weights(
        branch_weights,
        n_reference=n_reference,
        n_grid=n_grid,
        n_child=int(n_child_shocks),
        device=device,
        dtype=parent_states.dtype,
    )
    rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    default_boundary_status: Dict[str, str] = {}
    raw_tensors: Dict[str, Any] = {}
    checkpoint_metadata = []

    specs = _as_checkpoint_specs(
        checkpoint_paths,
        sdf_checkpoint=sdf_checkpoint,
        hyperparams_json=hyperparams_json,
        config_json=config_json,
        checkpoint_labels=checkpoint_labels,
    )
    loaded = [
        load_analysis_checkpoint_spec(
            spec,
            allow_default_hyperparams=allow_default_hyperparams,
            allow_current_config=allow_current_config,
            device=device,
        )
        for spec in specs
    ]
    labels = [ckpt.metadata["label"] for ckpt in loaded]
    if len(labels) != len(set(labels)):
        raise ValueError(f"checkpoint labels must be unique, got {labels}")
    models_for_state = [m for ckpt in loaded for m in (ckpt.models["policy_value"], ckpt.models["sdf_fc1"])]
    with preserve_analysis_state(models_for_state):
        for ckpt_idx, ckpt in enumerate(loaded):
            metadata = dict(ckpt.metadata)
            checkpoint_metadata.append(metadata)
            backend = VectorizedBellmanSurfaceBackend(
                ckpt.models["policy_value"],
                ckpt.hyperparams,
                economic_config=ckpt.economic_config,
            )
            per_metric_values: Dict[Tuple[str, str, str], List[torch.Tensor]] = {}
            # Use a canonical full parent batch to keep CRN and floating-point
            # evaluation independent of caller chunk-size choices.
            for start in range(0, parent_states.shape[0], parent_states.shape[0]):
                end = parent_states.shape[0]
                ps = parent_states[start:end]
                macro_chunk = MacroTransitionContext(
                    hatc_cal=macro.hatc_cal[start:end],
                    lnk_cal=macro.lnk_cal[start:end],
                )
                ref_idx_chunk = parent_reference_index[start:end]
                sb = shock_bank.gather(ref_idx_chunk)
                weights_chunk = None if weights_all is None else weights_all[start:end]
                child = build_child_exogenous_bundle(
                    ckpt.models["sdf_fc1"],
                    ps,
                    macro_chunk,
                    sb,
                    economic_config=ckpt.economic_config,
                    branch_weights=weights_chunk,
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

            checkpoint_label = str(metadata["label"])
            raw_tensors[checkpoint_label] = {}
            bb, zz = torch.meshgrid(b_grid_t.cpu(), z_grid_t.cpu(), indexing="xy")
            flat_b = bb.reshape(-1).numpy()
            flat_z = zz.reshape(-1).numpy()
            phat_np = default_np = survival_np = None
            if state_mode == FIXED_GRID_MODE:
                with torch.no_grad():
                    parent_out = ckpt.models["policy_value"](parent_states)
                    phat = _policy_get(parent_out, "Phat").detach().cpu().reshape(-1)
                    default_probability = _policy_get(parent_out, "bar_z").detach().cpu().reshape(-1)
                    survival_probability = (1.0 - default_probability).reshape(-1)
                phat_np = phat.numpy()
                default_np = default_probability.numpy()
                survival_np = survival_probability.numpy()
                phat_grid = phat_np.reshape(len(z_grid_t), len(b_grid_t))
                status = _fixed_boundary_status(phat_grid)
                default_boundary_status[checkpoint_label] = status
                boundary_rows.extend(
                    _extract_default_boundary_rows(
                        checkpoint=checkpoint_label,
                        b_values=b_grid_t.detach().cpu().numpy(),
                        z_values=z_grid_t.detach().cpu().numpy(),
                        phat_grid=phat_grid,
                    )
                )
            support_distance = support.get("distance")
            in_support = support.get("in_support")
            for key, chunks in per_metric_values.items():
                eq, mode, metric = key
                values = torch.cat(chunks, dim=0)
                raw_tensors[checkpoint_label][f"{eq}.{mode}.{metric}"] = values
                if state_mode == FIXED_GRID_MODE:
                    if metric != "conditional_signed" or mode != "train":
                        continue
                    signed_np = values.reshape(-1).numpy()
                    abs_np = np.abs(signed_np)
                    raw_tensors[checkpoint_label][f"{eq}.train.conditional_abs"] = torch.from_numpy(abs_np)
                    for pos, signed_val in enumerate(signed_np):
                        rows.append({
                            "checkpoint": checkpoint_label,
                            "checkpoint_hash": metadata.get("checkpoint_sha256"),
                            "policy_state_hash": metadata.get("policy_state_hash"),
                            "sdf_state_hash": metadata.get("sdf_state_hash"),
                            "equation": eq,
                            "m_mode": "train",
                            "aggregation": "none",
                            "b": float(flat_b[pos]),
                            "z": float(flat_z[pos]),
                            "conditional_signed": float(signed_val),
                            "conditional_abs": float(abs_np[pos]),
                            "phat": float(phat_np[pos]),
                            "default_probability": float(default_np[pos]),
                            "survival_probability": float(survival_np[pos]),
                            "n_child_shocks": int(n_child_shocks),
                            "seed": int(seed),
                            "eta": float(fixed_state["eta"]),
                            "i": float(fixed_state["i"]),
                            "x": float(fixed_state["x"]),
                            "hatcf": float(fixed_state["hatcf"]),
                            "lnkf": float(fixed_state["lnkf"]),
                            "hatc_cal": float(fixed_state["hatc_cal"]),
                            "lnk_cal": float(fixed_state["lnk_cal"]),
                            "state_mode": state_mode,
                        })
                    continue
                aggs = _aggregate_values(values, state_mode, n_grid, n_reference)
                finite = torch.isfinite(values.reshape(n_reference, n_grid) if state_mode != "fixed_slice" else values.reshape(1, n_grid))
                finite_ratio = finite.float().mean(dim=0).numpy()
                n_finite = finite.sum(dim=0).numpy()
                all_nonfinite = (~finite.any(dim=0)).numpy()
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
                            "all_nonfinite": bool(all_nonfinite[grid_pos]),
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
        "default_boundary_status": default_boundary_status,
        "workload_estimate": workload,
        "plotting": {
            "apply_support_mask": bool(apply_support_mask),
            "include_raw": bool(include_raw_plots),
            "include_signed": bool(include_signed),
            "residual_threshold": None if residual_threshold is None else float(residual_threshold),
            "log_residual_scale": bool(log_residual_scale),
        },
    }
    result = ConvergenceSurfaceResult(
        long_table=table,
        raw_tensors=raw_tensors,
        manifests=manifests,
        checkpoint_metadata=checkpoint_metadata,
        shock_bank_metadata={
            "seed": int(seed),
            "n_child_shocks": int(n_child_shocks),
            "common_random_numbers": True,
            "shock_bank_base_shape": list(shock_bank.base_shape),
            "shock_bank_expanded_shape": [int(parent_states.shape[0]), int(n_child_shocks), 1],
            "shock_bank_storage_numel": int(shock_bank.storage_numel),
            "shock_reuse_axis": "all_bz_grid_points_within_reference",
            "shock_bank_hash": _tensor_hash([shock_bank.eps_x, shock_bank.eps_z, shock_bank.u_eta, shock_bank.u_i]),
        },
        support_metadata=support,
        disabled_metrics={"policy_regret": "disabled_first_version"},
    )
    result.raw_tensors["default_boundary_rows"] = boundary_rows
    if output_dir is not None:
        save_convergence_surface_result(
            result,
            output_dir,
            include_raw=include_raw_plots,
            apply_support_mask=apply_support_mask,
        )
    return result


def save_convergence_surface_result(
    result: ConvergenceSurfaceResult,
    output_dir: str | Path,
    *,
    include_raw: bool = False,
    apply_support_mask: bool = True,
) -> None:
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
    boundary_rows = result.raw_tensors.get("default_boundary_rows", [])
    if boundary_rows:
        pd.DataFrame(boundary_rows).to_csv(output / "default_boundary.csv", index=False)
    else:
        pd.DataFrame(columns=["checkpoint", "component_id", "point_index", "b", "z"]).to_csv(
            output / "default_boundary.csv",
            index=False,
        )
    if result.manifests.get("state_mode") == FIXED_GRID_MODE:
        _write_fixed_grid_pngs(
            result.long_table,
            output,
            include_signed=bool(result.manifests.get("plotting", {}).get("include_signed", False)),
            residual_threshold=result.manifests.get("plotting", {}).get("residual_threshold"),
            log_residual_scale=bool(result.manifests.get("plotting", {}).get("log_residual_scale", False)),
        )
    else:
        _write_primary_pngs(result.long_table, output, include_raw=include_raw, apply_support_mask=apply_support_mask)
        _write_support_pngs(result.long_table, output)


def _write_fixed_grid_pngs(
    table: pd.DataFrame,
    output: Path,
    *,
    include_signed: bool = False,
    residual_threshold: Optional[float] = None,
    log_residual_scale: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    if table.empty:
        return
    b_values = np.array(sorted(table["b"].unique()), dtype=float)
    z_values = np.array(sorted(table["z"].unique()), dtype=float)
    for equation in ("p0", "pi", "q"):
        eq_all = table[table["equation"] == equation]
        if eq_all.empty:
            continue
        abs_vmax = np.nanmax(eq_all["conditional_abs"].to_numpy(dtype=float))
        if not np.isfinite(abs_vmax) or abs_vmax <= 0.0:
            abs_vmax = 1.0
        signed_vmax = np.nanmax(np.abs(eq_all["conditional_signed"].to_numpy(dtype=float)))
        if not np.isfinite(signed_vmax) or signed_vmax <= 0.0:
            signed_vmax = 1.0
        for checkpoint, sub in eq_all.groupby("checkpoint"):
            piv_abs = sub.pivot_table(index="z", columns="b", values="conditional_abs", aggfunc="first").sort_index()
            piv_signed = sub.pivot_table(index="z", columns="b", values="conditional_signed", aggfunc="first").sort_index()
            piv_phat = sub.pivot_table(index="z", columns="b", values="phat", aggfunc="first").sort_index()
            values = piv_abs.values
            color_label = "conditional_abs"
            if log_residual_scale:
                values = np.log10(values + 1e-12)
                color_label = "log10(conditional_abs + 1e-12)"
            fig, ax = plt.subplots(figsize=(6, 4.8))
            im = ax.imshow(
                values,
                origin="lower",
                aspect="auto",
                extent=[b_values.min(), b_values.max(), z_values.min(), z_values.max()],
                vmin=None if log_residual_scale else 0.0,
                vmax=None if log_residual_scale else abs_vmax,
            )
            _overlay_default_and_threshold(
                ax,
                b_values,
                z_values,
                piv_phat.values,
                piv_abs.values,
                residual_threshold=residual_threshold,
            )
            ax.set_xlabel("b")
            ax.set_ylabel("z")
            ax.set_title(f"{checkpoint} {equation} conditional_abs")
            fig.colorbar(im, ax=ax, label=color_label)
            fig.tight_layout()
            fig.savefig(output / f"{checkpoint}_{equation}_conditional_abs.png")
            plt.close(fig)

            if include_signed:
                fig, ax = plt.subplots(figsize=(6, 4.8))
                im = ax.imshow(
                    piv_signed.values,
                    origin="lower",
                    aspect="auto",
                    extent=[b_values.min(), b_values.max(), z_values.min(), z_values.max()],
                    vmin=-signed_vmax,
                    vmax=signed_vmax,
                    cmap="coolwarm",
                )
                _overlay_default_and_threshold(
                    ax,
                    b_values,
                    z_values,
                    piv_phat.values,
                    piv_abs.values,
                    residual_threshold=residual_threshold,
                )
                ax.set_xlabel("b")
                ax.set_ylabel("z")
                ax.set_title(f"{checkpoint} {equation} conditional_signed")
                fig.colorbar(im, ax=ax, label="conditional_signed")
                fig.tight_layout()
                fig.savefig(output / f"{checkpoint}_{equation}_conditional_signed.png")
                plt.close(fig)


def _overlay_default_and_threshold(
    ax: Any,
    b_values: np.ndarray,
    z_values: np.ndarray,
    phat_grid: np.ndarray,
    abs_grid: np.ndarray,
    *,
    residual_threshold: Optional[float],
) -> None:
    handles = []
    labels = []
    if _fixed_boundary_status(phat_grid) == "observed":
        cs = ax.contour(b_values, z_values, phat_grid, levels=[0.0], colors="black", linewidths=1.3)
        if cs.collections:
            handles.append(cs.collections[0])
            labels.append("default boundary: Phat=0")
    if residual_threshold is not None and np.nanmin(abs_grid) <= float(residual_threshold) <= np.nanmax(abs_grid):
        cs_thr = ax.contour(
            b_values,
            z_values,
            abs_grid,
            levels=[float(residual_threshold)],
            colors="white",
            linewidths=1.0,
            linestyles="dashed",
        )
        if cs_thr.collections:
            handles.append(cs_thr.collections[0])
            labels.append(f"conditional_abs={float(residual_threshold):g}")
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=8)


def _write_primary_pngs(
    table: pd.DataFrame,
    output: Path,
    *,
    include_raw: bool = False,
    apply_support_mask: bool = True,
) -> None:
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
            if apply_support_mask and "in_support" in sub.columns and sub["in_support"].notna().any():
                support = sub.pivot_table(index="z", columns="b", values="in_support", aggfunc="first").sort_index()
                pivot = pivot.where(support.astype(bool))
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
            fig.savefig(output / f"{checkpoint}_{equation}_{metric_name}_{mode}_{agg}.png")
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
                    vmax_abs = float(np.nanmax(np.abs(delta.values))) if np.isfinite(delta.values).any() else 0.0
                    if vmax_abs == 0.0:
                        vmax_abs = 1.0
                    fig, ax = plt.subplots(figsize=(5, 4))
                    im = ax.imshow(
                        delta.values,
                        origin="lower",
                        aspect="auto",
                        extent=[delta.columns.min(), delta.columns.max(), delta.index.min(), delta.index.max()],
                        vmin=-vmax_abs,
                        vmax=vmax_abs,
                        cmap="coolwarm",
                    )
                    ax.set_xlabel("b")
                    ax.set_ylabel("z")
                    ax.set_title(f"delta {other} minus {base} {equation} {metric}")
                    fig.colorbar(im, ax=ax, label=f"delta {metric}")
                    fig.tight_layout()
                    fig.savefig(output / f"delta_{other}_minus_{base}_{equation}_{metric_name}_{mode}_{agg}.png")
                    plt.close(fig)


def _write_support_pngs(table: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    if table.empty or "support_distance" not in table.columns:
        return
    base = table.drop_duplicates(["b", "z"])
    if base["support_distance"].isna().all():
        return
    distance = base.pivot_table(index="z", columns="b", values="support_distance", aggfunc="first").sort_index()
    mask = base.pivot_table(index="z", columns="b", values="in_support", aggfunc="first").sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        distance.values,
        origin="lower",
        aspect="auto",
        extent=[distance.columns.min(), distance.columns.max(), distance.index.min(), distance.index.max()],
    )
    ax.set_xlabel("b")
    ax.set_ylabel("z")
    ax.set_title("support distance")
    fig.colorbar(im, ax=ax, label="standardized nearest-reference distance")
    fig.tight_layout()
    fig.savefig(output / "support_distance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        mask.astype(float).values,
        origin="lower",
        aspect="auto",
        extent=[mask.columns.min(), mask.columns.max(), mask.index.min(), mask.index.max()],
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("b")
    ax.set_ylabel("z")
    ax.set_title("support mask")
    fig.colorbar(im, ax=ax, label="in support")
    fig.tight_layout()
    fig.savefig(output / "support_mask.png")
    plt.close(fig)
