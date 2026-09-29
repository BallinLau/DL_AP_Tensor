"""Read-only EP1/EP4 counterfactual decomposition of P continuation drift.

The evaluator holds a reference bank fixed while swapping the checkpoint SDF
pricing function and future-equity function.  It never constructs an optimizer,
runs backward, mutates a checkpoint, or changes the training path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

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
from evaluation.bp_diagnostics import (  # noqa: E402
    _checkpoint_economic_config,
    build_frozen_transition_data,
)
from evaluation.full_run_diagnostics import model_state_hash  # noqa: E402
from evaluation.grids import ReferenceFirmState, load_reference_state  # noqa: E402
from experiments.evaluate_p_teacher_drift import (  # noqa: E402
    _losses,
    _teacher_branch_components,
    atomic_write_csv,
    atomic_write_json,
    build_focus_states,
    choose_reference_artifacts,
    discover_checkpoints,
    select_branch_values,
)
from training.bp_grid_teacher import (  # noqa: E402
    BPGridTeacher,
    _expand_grid_children,
    _target_equity,
)


M_REPRODUCTION_TOL = 1e-6
PCHILD_REPRODUCTION_TOL = 1e-5
CONTINUATION_REPRODUCTION_TOL = 1e-5
SHAPLEY_IDENTITY_TOL = 1e-8
DEFAULT_Z_VALUES = (2.0, 3.0, 4.0)


@dataclass
class ReferenceBank:
    episode: int
    parent_states: torch.Tensor
    child_states: torch.Tensor
    child_weights: torch.Tensor
    original_m_raw: torch.Tensor
    original_m_used: torch.Tensor
    selected_is_pi: torch.Tensor
    bp_ref: torch.Tensor
    multiplier_ref: torch.Tensor
    teacher_p_child_mean: torch.Tensor
    teacher_continuation: torch.Tensor
    b: np.ndarray
    z: np.ndarray
    eta: np.ndarray
    i_index: np.ndarray
    i_value: np.ndarray
    selected_branch: np.ndarray
    shock_bank_sha256: str
    transition_metadata: Dict[str, Any]


def cube_column(m_episode: int, p_episode: int, r_episode: int) -> str:
    return f"F_M{int(m_episode)}_P{int(p_episode)}_R{int(r_episode)}"


def require_finite(name: str, values: Any) -> None:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        count = int((~np.isfinite(array)).sum())
        raise ValueError(f"{name} contains {count} nonfinite value(s)")


def contribution_ratio(effect: float, total: float) -> float:
    if not math.isfinite(effect) or not math.isfinite(total) or total == 0.0:
        return float("nan")
    return effect / total


def classify_mechanism(phi_p: float, phi_m: float, phi_r: float) -> str:
    magnitudes = {"P": abs(phi_p), "M": abs(phi_m), "R": abs(phi_r)}
    if magnitudes["P"] > 1.5 * max(magnitudes["M"], magnitudes["R"]):
        return "FUTURE_EQUITY_DOMINANT"
    if magnitudes["M"] > 1.5 * max(magnitudes["P"], magnitudes["R"]):
        return "SDF_DOMINANT"
    if magnitudes["R"] > 1.5 * max(magnitudes["P"], magnitudes["M"]):
        return "TRANSITION_POLICY_DOMINANT"
    return "MIXED"


def fixed_bank_shapley(
    cube: Mapping[str, float], *, old_episode: int, new_episode: int, bank_episode: int
) -> Dict[str, float]:
    a = float(cube[cube_column(old_episode, old_episode, bank_episode)])
    b = float(cube[cube_column(old_episode, new_episode, bank_episode)])
    c = float(cube[cube_column(new_episode, old_episode, bank_episode)])
    d = float(cube[cube_column(new_episode, new_episode, bank_episode)])
    p_effect = 0.5 * ((b - a) + (d - c))
    m_effect = 0.5 * ((c - a) + (d - b))
    return {
        "reference_bank": int(bank_episode),
        "total_model_change": d - a,
        "p_sequential_first": b - a,
        "m_sequential_second": d - b,
        "future_equity_effect": p_effect,
        "sdf_effect": m_effect,
        "identity_error": p_effect + m_effect - (d - a),
    }


def three_factor_shapley(
    cube: Mapping[str, float], *, old_episode: int, new_episode: int
) -> Dict[str, float]:
    value = lambda m, p, r: float(cube[cube_column(m, p, r)])
    v000 = value(old_episode, old_episode, old_episode)
    v100 = value(new_episode, old_episode, old_episode)
    v010 = value(old_episode, new_episode, old_episode)
    v001 = value(old_episode, old_episode, new_episode)
    v110 = value(new_episode, new_episode, old_episode)
    v101 = value(new_episode, old_episode, new_episode)
    v011 = value(old_episode, new_episode, new_episode)
    v111 = value(new_episode, new_episode, new_episode)
    phi_p = (
        (v010 - v000) / 3.0
        + (v110 - v100) / 6.0
        + (v011 - v001) / 6.0
        + (v111 - v101) / 3.0
    )
    phi_m = (
        (v100 - v000) / 3.0
        + (v110 - v010) / 6.0
        + (v101 - v001) / 6.0
        + (v111 - v011) / 3.0
    )
    phi_r = (
        (v001 - v000) / 3.0
        + (v101 - v100) / 6.0
        + (v011 - v010) / 6.0
        + (v111 - v110) / 3.0
    )
    total = v111 - v000
    return {
        "continuation_ep1": v000,
        "continuation_ep4": v111,
        "total_change": total,
        "future_equity_effect": phi_p,
        "sdf_effect": phi_m,
        "transition_policy_effect": phi_r,
        "future_equity_ratio": contribution_ratio(phi_p, total),
        "sdf_ratio": contribution_ratio(phi_m, total),
        "transition_policy_ratio": contribution_ratio(phi_r, total),
        "shapley_identity_error": phi_p + phi_m + phi_r - total,
        "mechanism_label": classify_mechanism(phi_p, phi_m, phi_r),
    }


def weighted_continuation(
    m_used: torch.Tensor,
    p_child: torch.Tensor,
    weights: torch.Tensor,
    multiplier: torch.Tensor,
) -> torch.Tensor:
    """Production continuation: g * sum_j w_j M_j P_j, with g applied once."""
    if m_used.shape != p_child.shape or m_used.shape != weights.shape:
        raise ValueError("m_used, p_child, and weights must have identical [N,J] shapes")
    if not bool(torch.isfinite(m_used).all() and torch.isfinite(p_child).all()):
        raise ValueError("continuation inputs contain nonfinite values")
    return multiplier.reshape(-1) * (weights * m_used * p_child).sum(dim=1)


def _tensor_max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    return float((left - right).abs().max().item())


def _selected(values0: torch.Tensor, valuesi: torch.Tensor, use_pi: torch.Tensor) -> torch.Tensor:
    return torch.where(use_pi.reshape(-1), valuesi.reshape(-1), values0.reshape(-1))


def _model_i_values(model: torch.nn.Module) -> np.ndarray:
    size = max(int(model.i_grid_size), 2)
    return np.linspace(0.0, float(model.i_threshold), size, dtype=np.float64)


def build_reference_bank(
    *,
    episode: int,
    loaded: Any,
    reference: ReferenceFirmState,
    b_value: float,
    z_values: Sequence[float],
    eta_value: float,
    n_child_shocks: int,
    shock_seed: int,
    device: torch.device,
) -> tuple[ReferenceBank, Dict[str, float]]:
    model = loaded.models["policy_value"]
    sdf_fc1 = loaded.models["sdf_fc1"]
    model.eval()
    sdf_fc1.eval()
    i_values = _model_i_values(model)
    base_states = build_focus_states(
        reference, [b_value], z_values, eta=eta_value,
        i_value=float(reference.i_mid), device=device,
    )
    transition = build_frozen_transition_data(
        sdf_fc1,
        base_states,
        reference,
        loaded.hyperparams,
        loaded.economic_config,
        n_child_shocks=n_child_shocks,
        shock_seed=shock_seed,
        shock_bank_max_child_shocks=n_child_shocks,
    )
    p0_loss, pi_loss = _losses(loaded)
    teacher = BPGridTeacher.from_hyperparams(model, p0_loss, pi_loss, loaded.hyperparams)
    expanded_count = int(transition.stacked_children().shape[1])
    parent_parts = []
    child_parts = []
    weight_parts = []
    m_raw_parts = []
    m_used_parts = []
    selected_parts = []
    bp_parts = []
    multiplier_parts = []
    teacher_p_parts = []
    teacher_cont_parts = []
    b_rows: list[float] = []
    z_rows: list[float] = []
    eta_rows: list[float] = []
    i_index_rows: list[int] = []
    i_value_rows: list[float] = []
    branch_rows: list[str] = []
    p_reproduction_errors: list[float] = []
    continuation_reproduction_errors: list[float] = []

    with _checkpoint_economic_config(loaded.economic_config), torch.no_grad():
        for i_index, i_value in enumerate(i_values):
            states = build_focus_states(
                reference, [b_value], z_values, eta=eta_value,
                i_value=float(i_value), device=device,
            )
            bundles = teacher.compute_multi_j_branches(
                [states, states],
                transition.stacked_children(),
                transition.stacked_m_used(),
                branches=["p0", "pi"],
                prefix_child_counts=[expanded_count],
                child_weights=transition.branch_weights,
            )
            result0 = bundles[0][expanded_count]
            resulti = bundles[1][expanded_count]
            comp0 = _teacher_branch_components(
                result0, states, branch="p0", p0_loss=p0_loss,
                pi_loss=pi_loss, model=model,
            )
            compi = _teacher_branch_components(
                resulti, states, branch="pi", p0_loss=p0_loss,
                pi_loss=pi_loss, model=model,
            )
            _, selected_np = select_branch_values(comp0["value"], compi["value"])
            selected_is_pi = torch.as_tensor(selected_np, device=device, dtype=torch.bool)
            bp0 = result0["bp_star"].reshape(-1)
            bpi = resulti["bp_star"].reshape(-1)
            bp_selected = _selected(bp0, bpi, selected_is_pi)
            multiplier = torch.where(
                selected_is_pi,
                torch.full_like(bp_selected, float(loaded.economic_config.G)),
                torch.ones_like(bp_selected),
            )
            selected_children, _ = _expand_grid_children(
                transition.stacked_children(),
                bp_selected.reshape(-1, 1),
                states[:, 0:1],
                states[:, 2:3],
            )
            selected_children = selected_children[:, 0]
            child_shape = selected_children.shape
            p_child, _ = _target_equity(model, selected_children.reshape(-1, child_shape[-1]))
            p_child = p_child.reshape(child_shape[0], child_shape[1])
            p_mean = (transition.branch_weights * p_child).sum(dim=1)
            teacher_p = torch.where(
                selected_is_pi,
                resulti["p_child_at_star"].reshape(-1),
                result0["p_child_at_star"].reshape(-1),
            )
            teacher_cont = torch.as_tensor(
                np.where(selected_np, compi["continuation"], comp0["continuation"]),
                device=device,
                dtype=states.dtype,
            )
            direct_cont = weighted_continuation(
                transition.stacked_m_used().squeeze(-1),
                p_child,
                transition.branch_weights,
                multiplier,
            )
            p_reproduction_errors.append(_tensor_max_abs(p_mean, teacher_p))
            continuation_reproduction_errors.append(_tensor_max_abs(direct_cont, teacher_cont))

            n_state = states.shape[0]
            parent_parts.append(states)
            child_parts.append(selected_children)
            weight_parts.append(transition.branch_weights)
            m_raw_parts.append(transition.stacked_m_raw().squeeze(-1))
            m_used_parts.append(transition.stacked_m_used().squeeze(-1))
            selected_parts.append(selected_is_pi)
            bp_parts.append(bp_selected)
            multiplier_parts.append(multiplier)
            teacher_p_parts.append(teacher_p)
            teacher_cont_parts.append(teacher_cont)
            for state_index, z_value in enumerate(z_values):
                b_rows.append(float(b_value))
                z_rows.append(float(z_value))
                eta_rows.append(float(eta_value))
                i_index_rows.append(int(i_index))
                i_value_rows.append(float(i_value))
                branch_rows.append("pi" if bool(selected_np[state_index]) else "p0")
            if n_state != len(z_values):
                raise AssertionError("focus-state ordering changed unexpectedly")

    bank = ReferenceBank(
        episode=int(episode),
        parent_states=torch.cat(parent_parts, dim=0),
        child_states=torch.cat(child_parts, dim=0),
        child_weights=torch.cat(weight_parts, dim=0),
        original_m_raw=torch.cat(m_raw_parts, dim=0),
        original_m_used=torch.cat(m_used_parts, dim=0),
        selected_is_pi=torch.cat(selected_parts, dim=0),
        bp_ref=torch.cat(bp_parts, dim=0),
        multiplier_ref=torch.cat(multiplier_parts, dim=0),
        teacher_p_child_mean=torch.cat(teacher_p_parts, dim=0),
        teacher_continuation=torch.cat(teacher_cont_parts, dim=0),
        b=np.asarray(b_rows, dtype=np.float64),
        z=np.asarray(z_rows, dtype=np.float64),
        eta=np.asarray(eta_rows, dtype=np.float64),
        i_index=np.asarray(i_index_rows, dtype=np.int64),
        i_value=np.asarray(i_value_rows, dtype=np.float64),
        selected_branch=np.asarray(branch_rows, dtype=object),
        shock_bank_sha256=str(transition.metadata["shock_bank_sha256"]),
        transition_metadata=dict(transition.metadata),
    )
    return bank, {
        "pchild_reproduction_abs_max": max(p_reproduction_errors, default=float("nan")),
        "teacher_continuation_reproduction_abs_max": max(
            continuation_reproduction_errors, default=float("nan")
        ),
    }


def recompute_m_on_bank(
    *, loaded: Any, bank: ReferenceBank, reference: ReferenceFirmState
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute production SDF on fixed child macro states without rerunning FC1."""
    sdf_fc1 = loaded.models["sdf_fc1"]
    sdf_fc1.eval()
    output_raw = torch.empty_like(bank.original_m_raw)
    # Match build_frozen_transition_data's numerical layout exactly. It computes
    # continuous shocks for one production parent batch, then duplicates each M
    # across the exact eta={0,1} pair. The bank repeats that parent batch over the
    # i-grid, so recomputing all i rows in one larger GEMM can differ by a few ulps.
    for i_index in np.unique(bank.i_index):
        positions_np = np.flatnonzero(bank.i_index == i_index)
        positions = torch.as_tensor(
            positions_np, device=bank.parent_states.device, dtype=torch.long
        )
        parent = bank.parent_states.index_select(0, positions)
        children_expanded = bank.child_states.index_select(0, positions)
        n_rows, expanded_children, _ = children_expanded.shape
        if expanded_children % 2 != 0:
            raise ValueError("exact-eta child bank must contain eta pairs")
        children = children_expanded[:, 0::2]
        paired = children_expanded[:, 1::2]
        if not torch.equal(children[..., [1, 3, 4, 5, 6]], paired[..., [1, 3, 4, 5, 6]]):
            raise ValueError("exact-eta child pairs differ outside eta")
        n_children = children.shape[1]
        dtype = parent.dtype
        device = parent.device
        parent_c = torch.full((n_rows, 1), reference.hatc_cal, device=device, dtype=dtype)
        parent_k = torch.full((n_rows, 1), reference.lnk_cal, device=device, dtype=dtype)
        parent_x = parent[:, 4:5]
        child_x = children[:, :, 4:5]
        child_c = children[:, :, 5:6]
        child_k = children[:, :, 6:7]
        with _checkpoint_economic_config(loaded.economic_config), torch.no_grad():
            w_parent = sdf_fc1.value_model(
                torch.cat([parent_x, parent_c, parent_k], dim=-1)
            )
            child_input = torch.cat([child_x, child_c, child_k], dim=-1)
            w_child = sdf_fc1.value_model(
                child_input.reshape(-1, 3)
            ).reshape(n_rows, n_children, 1)
            block_raw = sdf_fc1.sdf_model.get_M(
                w_parent=w_parent.squeeze(-1),
                w_children=w_child.squeeze(-1),
                k_parent=parent_k.squeeze(-1),
                k_children=child_k.squeeze(-1),
                c_parent=parent_c.squeeze(-1),
                c_children=child_c.squeeze(-1),
            )
        if isinstance(block_raw, list):
            block_raw = torch.stack(block_raw, dim=1)
        if block_raw.ndim == 3:
            block_raw = block_raw.squeeze(-1)
        output_raw.index_copy_(
            0, positions, block_raw.unsqueeze(2).expand(-1, -1, 2).reshape(n_rows, -1)
        )
    m_raw = output_raw
    use_clipped = bool(getattr(loaded.hyperparams, "pv_use_clipped_m", True))
    if use_clipped:
        m_used = m_raw.clamp(
            float(getattr(loaded.hyperparams, "pv_m_clamp_min", 0.7)),
            float(getattr(loaded.hyperparams, "pv_m_clamp_max", 1.3)),
        )
    else:
        m_used = m_raw
    return m_raw, m_used


def evaluate_pchild_on_bank(*, loaded: Any, bank: ReferenceBank) -> torch.Tensor:
    model = loaded.models["policy_value"]
    model.eval()
    n_rows, n_children, state_dim = bank.child_states.shape
    with _checkpoint_economic_config(loaded.economic_config), torch.no_grad():
        p_child, _bar_z = _target_equity(
            model, bank.child_states.reshape(n_rows * n_children, state_dim)
        )
    return p_child.reshape(n_rows, n_children)


def _per_i_frame(
    bank: ReferenceBank,
    *,
    m_by_episode: Mapping[int, torch.Tensor],
    p_by_episode: Mapping[int, torch.Tensor],
    old_episode: int,
    new_episode: int,
) -> pd.DataFrame:
    rows = []
    for row_index in range(len(bank.z)):
        row: Dict[str, Any] = {
            "b": bank.b[row_index],
            "z": bank.z[row_index],
            "eta": bank.eta[row_index],
            "i_index": bank.i_index[row_index],
            "i_value": bank.i_value[row_index],
            "reference_bank": bank.episode,
            "selected_branch_ref": bank.selected_branch[row_index],
            "bp_ref": float(bank.bp_ref[row_index].item()),
            "multiplier_ref": float(bank.multiplier_ref[row_index].item()),
        }
        for m_episode in (old_episode, new_episode):
            for p_episode in (old_episode, new_episode):
                row[f"F_M{m_episode}_P{p_episode}"] = float(
                    weighted_continuation(
                        m_by_episode[m_episode][row_index:row_index + 1],
                        p_by_episode[p_episode][row_index:row_index + 1],
                        bank.child_weights[row_index:row_index + 1],
                        bank.multiplier_ref[row_index:row_index + 1],
                    ).item()
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _child_frame(
    bank: ReferenceBank,
    *,
    m_raw: Mapping[int, torch.Tensor],
    m_used: Mapping[int, torch.Tensor],
    p_child: Mapping[int, torch.Tensor],
    old_episode: int,
    new_episode: int,
) -> pd.DataFrame:
    rows = []
    for row_index in range(len(bank.z)):
        for child_index in range(bank.child_states.shape[1]):
            m1 = float(m_used[old_episode][row_index, child_index].item())
            m4 = float(m_used[new_episode][row_index, child_index].item())
            p1 = float(p_child[old_episode][row_index, child_index].item())
            p4 = float(p_child[new_episode][row_index, child_index].item())
            rows.append({
                "b": bank.b[row_index],
                "z": bank.z[row_index],
                "eta": bank.eta[row_index],
                "i_index": bank.i_index[row_index],
                "i_value": bank.i_value[row_index],
                "child_index": child_index,
                "reference_bank_episode": bank.episode,
                "selected_branch_ref": bank.selected_branch[row_index],
                "selected_is_pi_ref": int(bank.selected_is_pi[row_index].item()),
                "bp_ref": float(bank.bp_ref[row_index].item()),
                "continuation_multiplier_ref": float(bank.multiplier_ref[row_index].item()),
                "child_weight": float(bank.child_weights[row_index, child_index].item()),
                f"M{old_episode}_raw": float(m_raw[old_episode][row_index, child_index].item()),
                f"M{old_episode}_used": m1,
                f"M{new_episode}_raw": float(m_raw[new_episode][row_index, child_index].item()),
                f"M{new_episode}_used": m4,
                f"Pchild_{old_episode}": p1,
                f"Pchild_{new_episode}": p4,
                f"M{old_episode}_P{old_episode}": m1 * p1,
                f"M{old_episode}_P{new_episode}": m1 * p4,
                f"M{new_episode}_P{old_episode}": m4 * p1,
                f"M{new_episode}_P{new_episode}": m4 * p4,
            })
    return pd.DataFrame(rows)


def build_cube(
    per_i_by_bank: Mapping[int, pd.DataFrame], *, old_episode: int, new_episode: int
) -> pd.DataFrame:
    rows = []
    for keys in sorted(
        set(
            (float(row.b), float(row.z), float(row.eta))
            for frame in per_i_by_bank.values()
            for row in frame[["b", "z", "eta"]].drop_duplicates().itertuples(index=False)
        )
    ):
        b_value, z_value, eta_value = keys
        row: Dict[str, Any] = {"b": b_value, "z": z_value, "eta": eta_value}
        for bank_episode in (old_episode, new_episode):
            frame = per_i_by_bank[bank_episode]
            subset = frame.loc[
                np.isclose(frame["b"], b_value)
                & np.isclose(frame["z"], z_value)
                & np.isclose(frame["eta"], eta_value)
            ]
            if subset.empty:
                raise ValueError(f"missing reference-bank rows for state {keys}, R{bank_episode}")
            for m_episode in (old_episode, new_episode):
                for p_episode in (old_episode, new_episode):
                    source = f"F_M{m_episode}_P{p_episode}"
                    row[cube_column(m_episode, p_episode, bank_episode)] = float(
                        subset[source].mean()
                    )
        require_finite("counterfactual cube row", list(row.values())[3:])
        rows.append(row)
    return pd.DataFrame(rows)


def build_shapley_tables(
    cube_frame: pd.DataFrame, *, old_episode: int, new_episode: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shapley_rows = []
    fixed_rows = []
    for record in cube_frame.to_dict("records"):
        cube = {key: value for key, value in record.items() if key.startswith("F_")}
        require_finite("counterfactual cube row", list(cube.values()))
        result = three_factor_shapley(
            cube, old_episode=old_episode, new_episode=new_episode
        )
        if abs(result["shapley_identity_error"]) >= SHAPLEY_IDENTITY_TOL:
            raise ValueError(
                f"Shapley identity failed at z={record['z']}: "
                f"{result['shapley_identity_error']}"
            )
        shapley_rows.append({"b": record["b"], "z": record["z"], "eta": record["eta"], **result})
        for bank_episode in (old_episode, new_episode):
            fixed = fixed_bank_shapley(
                cube,
                old_episode=old_episode,
                new_episode=new_episode,
                bank_episode=bank_episode,
            )
            fixed_rows.append({
                "b": record["b"], "z": record["z"], "eta": record["eta"], **fixed
            })
    return pd.DataFrame(shapley_rows), pd.DataFrame(fixed_rows)


def existing_data_audit(
    analysis_root: Path,
    *,
    old_episode: int,
    new_episode: int,
    b_value: float,
    z_values: Sequence[float],
    eta_value: float,
    g_by_episode: Mapping[int, float],
) -> pd.DataFrame:
    by_i_path = analysis_root / "tables" / "p_teacher_components_by_i.csv"
    selected_path = analysis_root / "tables" / "p_teacher_selected_branch.csv"
    if not by_i_path.is_file() or not selected_path.is_file():
        raise FileNotFoundError(
            f"existing P-teacher tables are required: {by_i_path}, {selected_path}"
        )
    by_i = pd.read_csv(by_i_path)
    selected = pd.read_csv(selected_path)
    rows = []
    for episode in (old_episode, new_episode):
        for z_value in z_values:
            mask = (
                (by_i["episode"] == episode)
                & np.isclose(by_i["b"], b_value)
                & np.isclose(by_i["z"], z_value)
                & np.isclose(by_i["eta"], eta_value)
            )
            group = by_i.loc[mask].copy()
            level = selected.loc[
                (selected["episode"] == episode)
                & np.isclose(selected["b"], b_value)
                & np.isclose(selected["z"], z_value)
                & np.isclose(selected["eta"], eta_value)
            ]
            if group.empty or len(level) != 1:
                raise ValueError(f"missing existing analysis rows for episode={episode}, z={z_value}")
            use_pi = group["selected_is_pi"].astype(bool).to_numpy()
            p_child = np.where(use_pi, group["p_child_pi"], group["p_child_p0"])
            multiplier = np.where(use_pi, float(g_by_episode[episode]), 1.0)
            e_mp = float(np.mean(group["continuation_selected"].to_numpy() / multiplier))
            e_p = float(np.mean(p_child))
            rows.append({
                "episode": episode,
                "b": b_value,
                "z": z_value,
                "eta": eta_value,
                "E_Pchild": e_p,
                "E_MP": e_mp,
                "M_eff": e_mp / e_p if e_p != 0.0 else float("nan"),
                "selected_continuation": float(level.iloc[0]["continuation_component"]),
            })
    return pd.DataFrame(rows)


def existing_analysis_child_shocks(analysis_root: Path) -> int | None:
    metadata_path = analysis_root / "metadata.json"
    if not metadata_path.is_file():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    value = payload.get("n_child_shocks")
    return None if value is None else int(value)


def write_figures(
    shapley: pd.DataFrame, fixed: pd.DataFrame, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    ordered = shapley.sort_values("z")
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x, ordered["continuation_ep1"], marker="o", label="EP1")
    ax.plot(x, ordered["continuation_ep4"], marker="o", label="EP4")
    ax.set_xticks(x, [f"z={z:g}" for z in ordered["z"]])
    ax.set_ylabel("selected continuation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "continuation_total_ep1_ep4.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    columns = [
        ("future_equity_effect", "future equity"),
        ("sdf_effect", "SDF"),
        ("transition_policy_effect", "transition-policy"),
    ]
    width = 0.24
    for index, (column, label) in enumerate(columns):
        ax.bar(x + (index - 1) * width, ordered[column], width=width, label=label)
    ax.set_xticks(x, [f"z={z:g}" for z in ordered["z"]])
    ax.set_ylabel("EP1 to EP4 Shapley contribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "continuation_shapley_contributions.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    labels = []
    p_values = []
    m_values = []
    for record in fixed.sort_values(["z", "reference_bank"]).to_dict("records"):
        labels.append(f"z={record['z']:g}\nR{int(record['reference_bank'])}")
        p_values.append(record["future_equity_effect"])
        m_values.append(record["sdf_effect"])
    positions = np.arange(len(labels))
    ax.bar(positions - 0.18, p_values, width=0.36, label="P effect")
    ax.bar(positions + 0.18, m_values, width=0.36, label="M effect")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("fixed-bank Shapley effect")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "fixed_bank_p_vs_m_effects.png", dpi=160)
    plt.close(fig)


def summary_markdown(
    shapley: pd.DataFrame,
    fixed: pd.DataFrame,
    diagnostics: Mapping[str, Any],
) -> str:
    lines = ["# P continuation counterfactual summary", ""]
    for row in shapley.sort_values("z").itertuples(index=False):
        lines.extend([
            f"## b={row.b:g}, z={row.z:g}, eta={row.eta:g}",
            "",
            f"- EP1 continuation: {row.continuation_ep1:.10g}",
            f"- EP4 continuation: {row.continuation_ep4:.10g}",
            f"- Total increase: {row.total_change:.10g}",
            f"- Future-equity Shapley contribution: {row.future_equity_effect:.10g} "
            f"(ratio {row.future_equity_ratio:.10g})",
            f"- SDF Shapley contribution: {row.sdf_effect:.10g} "
            f"(ratio {row.sdf_ratio:.10g})",
            f"- Transition-policy Shapley contribution: {row.transition_policy_effect:.10g} "
            f"(ratio {row.transition_policy_ratio:.10g})",
            f"- Mechanism label: `{row.mechanism_label}`",
            f"- Shapley identity error: {row.shapley_identity_error:.3e}",
            "",
        ])
    lines.extend(["## Fixed-bank diagnostics", ""])
    for row in fixed.sort_values(["z", "reference_bank"]).itertuples(index=False):
        lines.append(
            f"- z={row.z:g}, R{int(row.reference_bank)}: "
            f"P={row.future_equity_effect:.10g}, M={row.sdf_effect:.10g}, "
            f"total={row.total_model_change:.10g}, identity={row.identity_error:.3e}"
        )
    z4 = shapley.loc[np.isclose(shapley["z"], 4.0)]
    if not z4.empty:
        row = z4.iloc[0]
        lines.extend([
            "",
            "## z=4 focus",
            "",
            f"At z=4, the transition-policy effect is {row['transition_policy_effect']:.10g}; "
            "this factor jointly contains branch, bp, G multiplier, child-state, and FC1-transition changes.",
        ])
    lines.extend([
        "",
        "## Reproduction and invariance",
        "",
        f"- M reproduction max error R1: {diagnostics['m_reproduction_abs_max_R1']:.3e}",
        f"- M reproduction max error R4: {diagnostics['m_reproduction_abs_max_R4']:.3e}",
        f"- Pchild reproduction max error R1: {diagnostics['pchild_reproduction_abs_max_R1']:.3e}",
        f"- Pchild reproduction max error R4: {diagnostics['pchild_reproduction_abs_max_R4']:.3e}",
        f"- Same-bank continuation reproduction EP1: {diagnostics['continuation_reproduction_abs_error_EP1']:.3e}",
        f"- Same-bank continuation reproduction EP4: {diagnostics['continuation_reproduction_abs_error_EP4']:.3e}",
        f"- Existing-analysis reproduction status: {diagnostics['existing_analysis_reproduction_status']} "
        f"(analysis J={diagnostics['existing_analysis_n_child_shocks']})",
        f"- Existing-analysis continuation gap EP1: {diagnostics['existing_analysis_continuation_abs_gap_EP1']:.3e}",
        f"- Existing-analysis continuation gap EP4: {diagnostics['existing_analysis_continuation_abs_gap_EP4']:.3e}",
        f"- Policy/SDF state hashes unchanged: {diagnostics['model_hash_invariant']}",
        "",
        "`M_eff = E[MP] / E[P]` in the audit table is descriptive and is not interpreted as `E[M]`.",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--existing-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--old-episode", type=int, default=1)
    parser.add_argument("--new-episode", type=int, default=4)
    parser.add_argument("--b-value", type=float, default=0.05)
    parser.add_argument("--z-values", type=float, nargs="+", default=list(DEFAULT_Z_VALUES))
    parser.add_argument("--eta-value", type=float, default=1.0)
    parser.add_argument("--n-child-shocks", type=int, default=64)
    parser.add_argument("--shock-seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty; use --overwrite: {output}")
    for directory in (output / "tables", output / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    errors: list[Dict[str, Any]] = []
    try:
        if args.n_child_shocks < 2:
            raise ValueError("--n-child-shocks must be at least 2")
        if args.old_episode == args.new_episode:
            raise ValueError("old and new episodes must differ")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.set_device(device)
        run_root = args.run_root.expanduser().resolve()
        analysis_root = args.existing_analysis.expanduser().resolve()
        checkpoints = discover_checkpoints(run_root)
        for episode in (args.old_episode, args.new_episode):
            if episode not in checkpoints:
                raise FileNotFoundError(f"missing combined checkpoint for episode {episode}")
        reference_firm, reference_macro = choose_reference_artifacts(run_root)
        _frame, reference = load_reference_state(reference_firm, macro_path=reference_macro)
        loaded = {
            episode: load_analysis_checkpoint(checkpoints[episode], device=device)
            for episode in (args.old_episode, args.new_episode)
        }
        for item in loaded.values():
            item.models["policy_value"].eval()
            item.models["sdf_fc1"].eval()
        hashes_before = {
            f"policy_value_ep{episode}": model_state_hash(item.models["policy_value"])
            for episode, item in loaded.items()
        } | {
            f"sdf_fc1_ep{episode}": model_state_hash(item.models["sdf_fc1"])
            for episode, item in loaded.items()
        }

        banks: Dict[int, ReferenceBank] = {}
        bank_diagnostics: Dict[int, Dict[str, float]] = {}
        for episode in (args.old_episode, args.new_episode):
            banks[episode], bank_diagnostics[episode] = build_reference_bank(
                episode=episode,
                loaded=loaded[episode],
                reference=reference,
                b_value=args.b_value,
                z_values=args.z_values,
                eta_value=args.eta_value,
                n_child_shocks=args.n_child_shocks,
                shock_seed=args.shock_seed,
                device=device,
            )
        if banks[args.old_episode].shock_bank_sha256 != banks[args.new_episode].shock_bank_sha256:
            raise ValueError("R1 and R4 do not share the same exogenous shock bank")

        child_frames = []
        per_i_by_bank: Dict[int, pd.DataFrame] = {}
        m_reproduction: Dict[int, float] = {}
        for bank_episode, bank in banks.items():
            m_raw: Dict[int, torch.Tensor] = {}
            m_used: Dict[int, torch.Tensor] = {}
            p_child: Dict[int, torch.Tensor] = {}
            for model_episode in (args.old_episode, args.new_episode):
                m_raw[model_episode], m_used[model_episode] = recompute_m_on_bank(
                    loaded=loaded[model_episode], bank=bank, reference=reference
                )
                p_child[model_episode] = evaluate_pchild_on_bank(
                    loaded=loaded[model_episode], bank=bank
                )
                require_finite(f"M{model_episode} on R{bank_episode}", m_used[model_episode].cpu())
                require_finite(f"P{model_episode} on R{bank_episode}", p_child[model_episode].cpu())
            m_reproduction[bank_episode] = _tensor_max_abs(
                m_used[bank_episode], bank.original_m_used
            )
            if m_reproduction[bank_episode] >= M_REPRODUCTION_TOL:
                raise ValueError(
                    f"M reproduction failed for R{bank_episode}: {m_reproduction[bank_episode]}"
                )
            p_mean = (bank.child_weights * p_child[bank_episode]).sum(dim=1)
            p_error = _tensor_max_abs(p_mean, bank.teacher_p_child_mean)
            bank_diagnostics[bank_episode]["pchild_reproduction_abs_max"] = max(
                bank_diagnostics[bank_episode]["pchild_reproduction_abs_max"], p_error
            )
            if bank_diagnostics[bank_episode]["pchild_reproduction_abs_max"] >= PCHILD_REPRODUCTION_TOL:
                raise ValueError(
                    f"Pchild reproduction failed for R{bank_episode}: "
                    f"{bank_diagnostics[bank_episode]['pchild_reproduction_abs_max']}"
                )
            per_i_by_bank[bank_episode] = _per_i_frame(
                bank,
                m_by_episode=m_used,
                p_by_episode=p_child,
                old_episode=args.old_episode,
                new_episode=args.new_episode,
            )
            child_frames.append(
                _child_frame(
                    bank,
                    m_raw=m_raw,
                    m_used=m_used,
                    p_child=p_child,
                    old_episode=args.old_episode,
                    new_episode=args.new_episode,
                )
            )

        cube = build_cube(
            per_i_by_bank, old_episode=args.old_episode, new_episode=args.new_episode
        )
        shapley, fixed = build_shapley_tables(
            cube, old_episode=args.old_episode, new_episode=args.new_episode
        )
        audit = existing_data_audit(
            analysis_root,
            old_episode=args.old_episode,
            new_episode=args.new_episode,
            b_value=args.b_value,
            z_values=args.z_values,
            eta_value=args.eta_value,
            g_by_episode={
                episode: float(loaded[episode].economic_config.G)
                for episode in (args.old_episode, args.new_episode)
            },
        )
        continuation_errors: Dict[int, float] = {}
        existing_continuation_gaps: Dict[int, float] = {}
        analysis_j = existing_analysis_child_shocks(analysis_root)
        analysis_comparable = analysis_j == int(args.n_child_shocks)
        for episode in (args.old_episode, args.new_episode):
            same = cube_column(episode, episode, episode)
            bank = banks[episode]
            bank_reference = pd.DataFrame({
                "b": bank.b,
                "z": bank.z,
                "eta": bank.eta,
                "teacher_continuation": bank.teacher_continuation.detach().cpu().numpy(),
            }).groupby(["b", "z", "eta"], as_index=False)["teacher_continuation"].mean()
            same_bank = cube[["b", "z", "eta", same]].merge(
                bank_reference,
                on=["b", "z", "eta"],
                how="left",
                validate="one_to_one",
            )
            require_finite(
                f"same-bank continuation EP{episode}",
                same_bank[[same, "teacher_continuation"]],
            )
            continuation_errors[episode] = float(
                np.max(np.abs(same_bank[same] - same_bank["teacher_continuation"]))
            )
            continuation_errors[episode] = max(
                continuation_errors[episode],
                float(bank_diagnostics[episode]["teacher_continuation_reproduction_abs_max"]),
            )
            if continuation_errors[episode] >= CONTINUATION_REPRODUCTION_TOL:
                raise ValueError(
                    f"same-bank continuation reproduction failed for EP{episode}: "
                    f"{continuation_errors[episode]}"
                )
            existing_merged = cube[["b", "z", "eta", same]].merge(
                audit.loc[audit["episode"] == episode, ["b", "z", "eta", "selected_continuation"]],
                on=["b", "z", "eta"],
                how="left",
                validate="one_to_one",
            )
            require_finite(
                f"existing continuation audit EP{episode}",
                existing_merged[[same, "selected_continuation"]],
            )
            existing_continuation_gaps[episode] = float(
                np.max(
                    np.abs(
                        existing_merged[same] - existing_merged["selected_continuation"]
                    )
                )
            )
            if (
                analysis_comparable
                and existing_continuation_gaps[episode] >= CONTINUATION_REPRODUCTION_TOL
            ):
                raise ValueError(
                    f"existing-analysis continuation reproduction failed for EP{episode}: "
                    f"{existing_continuation_gaps[episode]}"
                )

        hashes_after = {
            f"policy_value_ep{episode}": model_state_hash(item.models["policy_value"])
            for episode, item in loaded.items()
        } | {
            f"sdf_fc1_ep{episode}": model_state_hash(item.models["sdf_fc1"])
            for episode, item in loaded.items()
        }
        hash_invariant = hashes_before == hashes_after
        if not hash_invariant:
            raise RuntimeError("model state hash changed during read-only evaluation")

        diagnostics = {
            "m_reproduction_abs_max_R1": m_reproduction[args.old_episode],
            "m_reproduction_abs_max_R4": m_reproduction[args.new_episode],
            "pchild_reproduction_abs_max_R1": bank_diagnostics[args.old_episode]["pchild_reproduction_abs_max"],
            "pchild_reproduction_abs_max_R4": bank_diagnostics[args.new_episode]["pchild_reproduction_abs_max"],
            "continuation_reproduction_abs_error_EP1": continuation_errors[args.old_episode],
            "continuation_reproduction_abs_error_EP4": continuation_errors[args.new_episode],
            "existing_analysis_n_child_shocks": analysis_j,
            "existing_analysis_comparable_child_count": analysis_comparable,
            "existing_analysis_continuation_abs_gap_EP1": existing_continuation_gaps[
                args.old_episode
            ],
            "existing_analysis_continuation_abs_gap_EP4": existing_continuation_gaps[
                args.new_episode
            ],
            "existing_analysis_reproduction_status": (
                "strict_pass" if analysis_comparable else "not_comparable_child_count"
            ),
            "model_hash_invariant": hash_invariant,
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
        }
        child_level = pd.concat(child_frames, ignore_index=True)
        by_i = pd.concat(per_i_by_bank.values(), ignore_index=True)
        atomic_write_csv(output / "tables" / "continuation_counterfactual_child_level.csv", child_level)
        atomic_write_csv(output / "tables" / "continuation_counterfactual_by_i.csv", by_i)
        atomic_write_csv(output / "tables" / "continuation_counterfactual_cube.csv", cube)
        atomic_write_csv(output / "tables" / "continuation_shapley_ep1_ep4.csv", shapley)
        atomic_write_csv(output / "tables" / "fixed_bank_p_vs_m_effects.csv", fixed)
        atomic_write_csv(output / "tables" / "existing_continuation_audit.csv", audit)
        atomic_write_csv(output / "errors.csv", pd.DataFrame(columns=["block", "error", "traceback"]))
        atomic_write_json(output / "metadata.json", {
            "run_root": run_root,
            "existing_analysis": analysis_root,
            "old_episode": args.old_episode,
            "new_episode": args.new_episode,
            "b_value": args.b_value,
            "z_values": args.z_values,
            "eta_value": args.eta_value,
            "n_child_shocks": args.n_child_shocks,
            "shock_seed": args.shock_seed,
            "shock_bank_sha256": banks[args.old_episode].shock_bank_sha256,
            "i_grid_size_by_episode": {
                str(episode): int(loaded[episode].models["policy_value"].i_grid_size)
                for episode in (args.old_episode, args.new_episode)
            },
            "i_threshold_by_episode": {
                str(episode): float(loaded[episode].models["policy_value"].i_threshold)
                for episode in (args.old_episode, args.new_episode)
            },
            "m_semantics_by_episode": {
                str(episode): banks[episode].transition_metadata["m_mode"]
                for episode in (args.old_episode, args.new_episode)
            },
            "pchild_semantics": "PolicyValue.cal_phats hard P=max(Phat,0); no extra survival gate",
            "reference": reference.to_dict(),
            "diagnostics": diagnostics,
            "read_only": True,
        })
        write_figures(shapley, fixed, output / "figures")
        (output / "continuation_counterfactual_summary.md").write_text(
            summary_markdown(shapley, fixed, diagnostics), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        errors.append({
            "block": "evaluation",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
        })
        atomic_write_csv(output / "errors.csv", pd.DataFrame(errors))
        raise


if __name__ == "__main__":
    main()
