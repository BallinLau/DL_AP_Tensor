"""Value / policy surface and boundary evaluator for the convergence evaluator.

This module turns one checkpoint + one fixed bank entry into the *formal*
policy evidence of the firm side:

* **Grid BP policy**: the formal leverage policy is the Grid-BP greedy solution
  of ``training.bp_grid_teacher.BPGridTeacher`` (``compute`` /
  ``compute_value_target``) -- ``bp0_star`` / ``bpI_star`` and the executed
  leverage ``b_next`` under the financing gate ``b_next = eta_t * bp + (1 -
  eta_t) * b_t``.  The ``PolicyValueModel`` bp0/bpI/bp heads are reported as
  *compatibility* columns only (and their regret against the grid star
  quantifies how far they are from the formal policy).
* **Investment boundary**: zeros of ``PI(b, z, eta, i) - P0(b, z, eta, i)`` along
  the investment-cost axis, solved with
  ``evaluation.firm_surfaces.evaluate_investment_cutoff`` and
  ``evaluation.boundaries.zero_crossings``.  No-uniqueness is *reported*, never
  assumed: no-crossing states are classified always-invest / never-invest and
  multi-crossing states expose all crossing intervals.
* **Default boundary**: zeros of ``Phat(b, z, eta, m) = 0`` along z
  (``evaluation.boundaries.extract_phat_default_boundary``), classified
  single / multiple / all_default / all_survival / nonfinite.  ``bar_z`` is a
  *soft default probability* (``1 - sigmoid(Phat/tau_z)``), never a productivity
  cutoff; the conditional default probability reported here is computed from the
  conditional productivity transition distribution (fixed bank shocks), not
  renamed from ``bar_z``.
* **Bar_i_model vs Bar_i_executed**: the model's investment probability
  (``bar_i_cond``) and the executed investment action (``bar_i_eff``; forced 0
  at entry/birth nodes in the panel) are kept apart.
* **P04 objective slices**: the full Bellman RHS as a function of the candidate
  bp at representative states, with the issuance ``Q_claim`` overlaid (data is
  returned; plotting is done by the figures module).
* **Cross-episode drift**: :func:`compare_policy_across_episodes` reports
  ``bp_star_drift`` (eta = 1 rows) plus policy / default / investment flip
  rates on the fixed bank states.

Everything runs ``model.eval()`` + ``torch.inference_mode()``; no optimizer is
constructed (``optimizer_steps`` is 0) and the caller can wrap any evaluation in
``checkpoint_io.param_hash_guard``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from evaluation.boundaries import (
    compare_hard_soft_default_boundaries,
    extract_phat_default_boundary,
    zero_crossings,
)
from evaluation.firm_surfaces import (
    evaluate_firm_surfaces,
    evaluate_investment_cutoff,
    investment_margin_diagnostics,
)
from evaluation.grids import FrozenFirmGrid, ReferenceFirmState, build_frozen_grid
from utils.firm_transition import apply_refinancing_policy

from .checkpoint_io import EvalCheckpoint, semantic_record
from .eval_bank import CheckpointTransition, build_transition_for_checkpoint, content_sha256
from .firm_bellman import (
    _bank_payload_on_device,
    _evaluate_grid_chunked,
    _forward_fields,
    _module_device,
    build_grid_teacher,
    iter_row_chunks,
    q_issue_claim_at_candidates,
    resolve_teacher_target_model,
    source_labels,
    split_transition,
)


# Investment status of ``evaluate_investment_cutoff`` mapped to the protocol's
# always/never-invest wording (both columns are kept in the tables).
INVESTMENT_STATUS_LABELS = {
    "all_invest": "always_invest",
    "all_no_invest": "never_invest",
    "single_crossing": "single_crossing",
    "multiple_crossings": "multiple_crossings",
    "nonfinite": "nonfinite",
}
INVESTMENT_STATUS_VALUES = tuple(INVESTMENT_STATUS_LABELS.values())
DEFAULT_STATUS_VALUES = (
    "single_crossing",
    "multiple_crossings",
    "all_default",
    "all_survival",
    "no_identified_crossing",
    "nonfinite",
)


@dataclass
class PolicyEvalConfig:
    """Configuration of one value/policy surface + boundary evaluation run."""

    b_points: int = 21
    z_points: int = 21
    b_min: Optional[float] = None
    b_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    i_points: int = 51
    i_min: float = 0.0
    i_max: Optional[float] = None
    chunk_size: int = 8192
    eta_views: Tuple[float, float] = (0.0, 1.0)
    parent_chunk_size: int = 256
    teacher_max_expanded_states: Optional[int] = None
    teacher_target: str = "policy_value"
    objective_slice_points: int = 21
    representative_count: int = 9
    bp_match_tol: float = 1e-3
    policy_flip_tol: float = 1e-2
    q_boundary_phat_eps: float = 1e-2

    def __post_init__(self) -> None:
        if int(self.b_points) < 2 or int(self.z_points) < 2:
            raise ValueError("b_points and z_points must be at least 2")
        if int(self.i_points) < 2:
            raise ValueError("i_points must be at least 2")
        if int(self.objective_slice_points) < 2:
            raise ValueError("objective_slice_points must be at least 2")


@dataclass
class PolicyEvalResult:
    """Result of :func:`evaluate_firm_policy`.

    * ``metrics`` -- flat float metrics (``bp_*`` / ``grid_*`` / ``eta*_gate_*``
      / ``investment_*`` / ``default_*`` / ``objective_slice_*``).
    * ``tables`` -- ``policy_rows``, ``investment_boundary``, ``default_boundary``,
      ``hard_soft_boundary``, ``objective_slices``.
    * ``surfaces`` -- b x z arrays per eta view (``eta0_*`` / ``eta1_*``).
    * ``diagnostics`` -- caliber notes, teacher configuration, state hash.
    """

    metrics: Dict[str, float]
    tables: Dict[str, pd.DataFrame]
    surfaces: Dict[str, np.ndarray]
    diagnostics: Dict[str, Any]


def build_policy_reference(
    bank: Mapping[str, Any],
    *,
    eta: float,
    config: PolicyEvalConfig,
) -> ReferenceFirmState:
    """Reference firm state derived from the fixed bank rows.

    Mirrors ``evaluation.grids.ReferenceFirmState`` semantics (i quantiles
    10/50/90, remaining state columns at their bank medians) so surfaces stay
    comparable with the checkpoint evaluator artifacts; the belief columns
    ``Hatcf``/``LnKF`` stay in the state and ``hatc_cal``/``lnk_cal`` stay in the
    macro context (they are never mixed).
    """
    states = np.asarray(
        bank["parent_states"].detach().cpu()
        if torch.is_tensor(bank["parent_states"])
        else bank["parent_states"],
        dtype=np.float64,
    )
    if states.ndim != 2 or states.shape[1] != 7:
        raise ValueError("bank parent_states must have shape [N, 7]")
    i_values = states[:, 3]
    hatc_cal = np.asarray(
        bank["hatc_cal"].detach().cpu() if torch.is_tensor(bank["hatc_cal"]) else bank["hatc_cal"],
        dtype=np.float64,
    ).reshape(-1)
    lnk_cal = np.asarray(
        bank["lnk_cal"].detach().cpu() if torch.is_tensor(bank["lnk_cal"]) else bank["lnk_cal"],
        dtype=np.float64,
    ).reshape(-1)
    return ReferenceFirmState(
        eta=float(eta),
        i_low=float(np.quantile(i_values, 0.10)),
        i_mid=float(np.quantile(i_values, 0.50)),
        i_high=float(np.quantile(i_values, 0.90)),
        x=float(np.median(states[:, 4])),
        hatcf=float(np.median(states[:, 5])),
        lnkf=float(np.median(states[:, 6])),
        hatc_cal=float(np.median(hatc_cal)),
        lnk_cal=float(np.median(lnk_cal)),
        n_parent_rows=int(states.shape[0]),
        source="evaluation-bank-median",
        macro_source="evaluation-bank-median",
    )


def build_policy_grid(
    bank: Mapping[str, Any],
    reference: ReferenceFirmState,
    *,
    config: PolicyEvalConfig,
    device: torch.device | str = "cpu",
) -> FrozenFirmGrid:
    """b x z grid spanning the bank's leverage/productivity support."""
    states = np.asarray(
        bank["parent_states"].detach().cpu()
        if torch.is_tensor(bank["parent_states"])
        else bank["parent_states"],
        dtype=np.float64,
    )
    b_min = float(config.b_min) if config.b_min is not None else float(states[:, 0].min())
    b_max = float(config.b_max) if config.b_max is not None else float(states[:, 0].max())
    z_min = float(config.z_min) if config.z_min is not None else float(states[:, 1].min())
    z_max = float(config.z_max) if config.z_max is not None else float(states[:, 1].max())
    if not b_min < b_max:
        b_min, b_max = min(b_min, b_max) - 0.5, max(b_min, b_max) + 0.5
    if not z_min < z_max:
        z_min, z_max = min(z_min, z_max) - 0.5, max(z_min, z_max) + 0.5
    return build_frozen_grid(
        reference,
        b_min=b_min,
        b_max=b_max,
        b_points=int(config.b_points),
        z_min=z_min,
        z_max=z_max,
        z_points=int(config.z_points),
        device=torch.device(device),
    )


def grid_policy_solution(
    cp: EvalCheckpoint,
    transition: CheckpointTransition,
    *,
    config: PolicyEvalConfig,
    model: Optional[torch.nn.Module] = None,
    teacher=None,
    source: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Formal Grid-BP policy on the bank rows (one row per bank state).

    Solves ``bp0_star`` (no-invest branch) and ``bpI_star`` (invest branch) with
    the production Grid-BP teacher and applies the parent-eta financing gate to
    obtain the executed ``b_next``.  The bp head outputs are recorded as
    ``*_head`` compatibility columns together with their regret against the
    grid star -- they are *never* used for ``b_next``.
    """
    model = model if model is not None else cp.models["policy_value"]
    teacher = teacher if teacher is not None else build_grid_teacher(
        cp,
        target_model=resolve_teacher_target_model(cp, teacher_target=config.teacher_target),
        max_expanded_states=config.teacher_max_expanded_states,
    )
    device = _module_device(model)
    parent = transition.parent_states.to(device)
    children_list, m_used, _m_raw, weights = split_transition(transition, device=device)
    n_rows = int(parent.shape[0])
    fields = _forward_fields(
        model,
        parent,
        ("bp0", "bpI", "bar_i", "bar_i_cond", "bar_i_eff", "P0", "PI", "P", "Phat", "bar_z"),
        chunk_size=config.chunk_size,
    )

    per_branch: Dict[str, List[Dict[str, torch.Tensor]]] = {"p0": [], "pi": []}
    for start, stop in iter_row_chunks(n_rows, config.parent_chunk_size):
        chunk = slice(start, stop)
        for branch, head_bp in (("p0", fields["bp0"]), ("pi", fields["bpI"])):
            result = teacher.compute(
                parent[chunk].to(device),
                [item[chunk].to(device) for item in children_list],
                m_used[chunk].to(device),
                branch=branch,
                bp_pred=head_bp[chunk],
                child_weights=weights[chunk].to(device),
            )
            per_branch[branch].append(result)

    def _merged_tensor(branch: str, key: str) -> torch.Tensor:
        return (
            torch.cat([part[key] for part in per_branch[branch]], dim=0)
            .reshape(n_rows, -1)[:, 0]
            .detach()
        )

    def _merged(branch: str, key: str) -> np.ndarray:
        return (
            torch.cat([part[key] for part in per_branch[branch]], dim=0)
            .reshape(n_rows, -1)[:, 0]
            .detach()
            .to(torch.float64)
            .cpu().numpy()
        )

    def _merged_full(branch: str, key: str) -> np.ndarray:
        return (
            torch.cat([part[key] for part in per_branch[branch]], dim=0)
            .detach()
            .to(torch.float64)
            .cpu().numpy()
        )

    b = parent[:, 0:1]
    eta = parent[:, 2:3]
    bp0_star = _merged_tensor("p0", "bp_star").to(torch.float64).reshape(n_rows, 1)
    bpi_star = _merged_tensor("pi", "bp_star").to(torch.float64).reshape(n_rows, 1)
    bar_i = fields["bar_i"].reshape(n_rows, 1)
    bp_star_mix = bar_i * bpi_star + (1.0 - bar_i) * bp0_star
    b_next_p0 = apply_refinancing_policy(b, bp0_star, eta).reshape(-1)
    b_next_pi = apply_refinancing_policy(b, bpi_star, eta).reshape(-1)
    b_next_mix = apply_refinancing_policy(b, bp_star_mix, eta).reshape(-1)

    parent_np = parent.detach().to(torch.float64).cpu().numpy()
    rows = {
        "row_index": np.arange(n_rows, dtype=np.int64),
        "b": parent_np[:, 0],
        "z": parent_np[:, 1],
        "eta": parent_np[:, 2],
        "source_label": list(source) if source is not None else ["unknown"] * n_rows,
        "refi_active": _merged("p0", "refi_active"),
        "bp0_star": bp0_star.reshape(-1).cpu().numpy(),
        "bpi_star": bpi_star.reshape(-1).cpu().numpy(),
        "bp_star_mix": bp_star_mix.reshape(-1).cpu().numpy(),
        "b_next_p0": b_next_p0.cpu().numpy(),
        "b_next_pi": b_next_pi.cpu().numpy(),
        "b_next_mix": b_next_mix.cpu().numpy(),
        "bp0_head": fields["bp0"].reshape(-1).detach().to(torch.float64).cpu().numpy(),
        "bpI_head": fields["bpI"].reshape(-1).detach().to(torch.float64).cpu().numpy(),
        "bar_i_model": fields["bar_i_cond"].reshape(-1).detach().to(torch.float64).cpu().numpy(),
        "bar_i_executed": fields["bar_i_eff"].reshape(-1).detach().to(torch.float64).cpu().numpy(),
        "value_star_p0": _merged("p0", "value_star"),
        "value_star_pi": _merged("pi", "value_star"),
        "value_pred_p0_head": _merged("p0", "value_pred"),
        "value_pred_pi_head": _merged("pi", "value_pred"),
        "regret_p0_head": _merged("p0", "regret"),
        "regret_pi_head": _merged("pi", "regret"),
        "margin_p0": _merged("p0", "top2_margin"),
        "margin_pi": _merged("pi", "top2_margin"),
        "confidence_p0": _merged("p0", "confidence"),
        "confidence_pi": _merged("pi", "confidence"),
        "boundary_low_p0": _merged("p0", "boundary_low"),
        "boundary_high_p0": _merged("p0", "boundary_high"),
        "boundary_low_pi": _merged("pi", "boundary_low"),
        "boundary_high_pi": _merged("pi", "boundary_high"),
        "bp_star_grid_p0": _merged("p0", "bp_star_grid"),
        "bp_star_grid_pi": _merged("pi", "bp_star_grid"),
        "q_issue_at_star_p0": _merged("p0", "q_issue_at_star"),
        "q_issue_at_star_pi": _merged("pi", "q_issue_at_star"),
        "p_child_at_star_p0": _merged("p0", "p_child_at_star"),
        "p_child_at_star_pi": _merged("pi", "p_child_at_star"),
        "default_at_star_p0": _merged("p0", "default_at_star"),
        "default_at_star_pi": _merged("pi", "default_at_star"),
        "eta_next_active_share": _merged("p0", "eta_next_active_share"),
        "child_b_mean": _merged("p0", "child_b_mean"),
        "P0": fields["P0"].reshape(-1).detach().to(torch.float64).cpu().numpy(),
        "PI": fields["PI"].reshape(-1).detach().to(torch.float64).cpu().numpy(),
        "P": fields["P"].reshape(-1).detach().to(torch.float64).cpu().numpy(),
        "Phat": fields["Phat"].reshape(-1).detach().to(torch.float64).cpu().numpy(),
        "bar_z": fields["bar_z"].reshape(-1).detach().to(torch.float64).cpu().numpy(),
    }
    return pd.DataFrame(rows), {
        "teacher_grid": {
            "grid_min": teacher.grid_min,
            "grid_max": teacher.grid_max,
            "coarse_size": teacher.coarse_size,
            "fine_size": teacher.fine_size,
            "refine": teacher.refine,
        }
    }


def solve_investment_boundary(
    model: torch.nn.Module,
    grid: FrozenFirmGrid,
    reference: ReferenceFirmState,
    *,
    config: PolicyEvalConfig,
    survival_mask: np.ndarray,
    i_max: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, np.ndarray]]:
    """Investment boundary: zeros of ``PI - V0`` along the investment-cost axis.

    No-uniqueness is reported, never assumed: states without a crossing are
    labelled always-invest / never-invest, states with several crossings expose
    every crossing value and interval, and the ``i_star`` column is only filled
    for exactly one crossing (and only inside the survival mask).
    """
    upper = float(i_max) if i_max is not None else float(model.i_threshold)
    investment = evaluate_investment_cutoff(
        model,
        grid,
        reference,
        i_points=int(config.i_points),
        i_min=float(config.i_min),
        i_max=upper,
        chunk_size=int(config.chunk_size),
        survival_mask=np.asarray(survival_mask, dtype=bool),
    )
    i_values = np.asarray(investment["i_values"], dtype=np.float64)
    delta = np.asarray(investment["delta_vi_v0"], dtype=np.float64)
    status_raw = np.asarray(investment["investment_status"], dtype=object)
    status_label = np.asarray(
        [INVESTMENT_STATUS_LABELS.get(str(value), str(value)) for value in status_raw.reshape(-1)],
        dtype=object,
    ).reshape(status_raw.shape)
    survival = np.asarray(investment["survival_mask"], dtype=bool)

    rows: List[Dict[str, Any]] = []
    n_b, n_z = grid.shape
    for b_pos in range(n_b):
        for z_pos in range(n_z):
            flat = b_pos * n_z + z_pos
            y = delta[b_pos, z_pos]
            crossings = zero_crossings(i_values, np.asarray(y, dtype=np.float64))
            intervals = [
                f"[{float(left):.6g},{float(right):.6g}]"
                for left, right in zip(crossings[:-1], crossings[1:])
            ]
            rows.append(
                {
                    "b": float(grid.b_values[b_pos]),
                    "z": float(grid.z_values[z_pos]),
                    "eta_view": float(reference.eta),
                    "i_star": float(investment["i_star"][b_pos, z_pos])
                    if np.isfinite(investment["i_star"][b_pos, z_pos])
                    else float("nan"),
                    "investment_status": str(status_raw.reshape(-1)[flat]),
                    "investment_status_label": str(status_label.reshape(-1)[flat]),
                    "crossing_count": int(len(crossings)),
                    "crossings_json": json.dumps([float(value) for value in crossings]),
                    "crossing_intervals_json": json.dumps(intervals),
                    "survival": bool(survival.reshape(-1)[flat]),
                }
            )
    frame = pd.DataFrame(rows)

    metrics: Dict[str, float] = {}
    for scope, mask in (("raw", np.ones_like(survival, dtype=bool)), ("survival", survival)):
        total = max(int(mask.sum()), 1)
        for value in INVESTMENT_STATUS_VALUES:
            metrics[f"investment_{value}_share_{scope}"] = float(
                (status_label[mask] == value).sum()
            ) / float(total)
    i_star = np.asarray(investment["i_star"], dtype=np.float64)
    i_star_survival = i_star[survival]
    finite_star = i_star_survival[np.isfinite(i_star_survival)]
    metrics["investment_i_star_mean_survival"] = (
        float(finite_star.mean()) if finite_star.size else float("nan")
    )
    metrics["investment_i_star_p90_survival"] = (
        float(np.quantile(finite_star, 0.9)) if finite_star.size else float("nan")
    )
    metrics["investment_multi_crossing_rows"] = float(
        (status_label.reshape(-1) == "multiple_crossings").sum()
    )
    metrics["investment_crossing_count_total"] = float(
        frame["crossing_count"].to_numpy().sum()
    )
    arrays = {
        "i_values": i_values,
        "delta_vi_v0": delta,
        "i_star": i_star,
        "investment_status": status_label,
        "investment_status_raw": status_raw,
        "survival_mask": survival,
    }
    return frame, metrics, arrays


def solve_default_boundary(
    grid: FrozenFirmGrid,
    phat: np.ndarray,
    bar_z: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, np.ndarray]]:
    """Default boundary: zeros of ``Phat(b, z) = 0`` along z.

    Every crossing is reported (the single ``z_default`` column follows the
    ``extract_phat_default_boundary`` contract and is NaN for multi-crossing
    rows); ``bar_z`` is only used for the soft/hard comparison and is never
    reinterpreted as a productivity cutoff.
    """
    phat_arr = np.asarray(phat, dtype=np.float64)
    bar_z_arr = np.asarray(bar_z, dtype=np.float64)
    frame, summary = extract_phat_default_boundary(
        np.asarray(grid.b_values), np.asarray(grid.z_values), phat_arr
    )
    crossings_all = []
    intervals_all = []
    for b_pos in range(len(grid.b_values)):
        crossings = zero_crossings(np.asarray(grid.z_values), phat_arr[b_pos])
        crossings_all.append(json.dumps([float(value) for value in crossings]))
        intervals_all.append(
            json.dumps(
                [
                    f"[{float(left):.6g},{float(right):.6g}]"
                    for left, right in zip(crossings[:-1], crossings[1:])
                ]
            )
        )
    frame["crossings_json"] = crossings_all
    frame["crossing_intervals_json"] = intervals_all

    hard_soft, hard_soft_summary = compare_hard_soft_default_boundaries(
        np.asarray(grid.b_values), np.asarray(grid.z_values), phat_arr, bar_z_arr
    )

    metrics: Dict[str, float] = {}
    n_rows = max(int(len(frame)), 1)
    for value in DEFAULT_STATUS_VALUES:
        metrics[f"default_{value}_share"] = float(
            (frame["boundary_status"] == value).sum()
        ) / float(n_rows)
    metrics.update(
        {
            "default_boundary_single_crossing_share": summary[
                "default_boundary_single_crossing_share"
            ],
            "default_boundary_multiple_crossing_share": summary[
                "default_boundary_multiple_crossing_share"
            ],
            "default_boundary_monotonic_share": summary["default_boundary_monotonic_share"],
            "default_boundary_roughness": summary["default_boundary_roughness"],
            "hard_default_share": hard_soft_summary["hard_default_share"],
            "soft_default_mean": hard_soft_summary["soft_default_mean"],
            "hard_soft_boundary_comparable_share": hard_soft_summary[
                "hard_soft_boundary_comparable_share"
            ],
            "hard_soft_boundary_abs_gap_mean": hard_soft_summary[
                "hard_soft_boundary_abs_gap_mean"
            ],
            "hard_soft_boundary_abs_gap_p90": hard_soft_summary[
                "hard_soft_boundary_abs_gap_p90"
            ],
        }
    )
    arrays = {
        "z_default": frame["z_default"].to_numpy(dtype=np.float64),
        "boundary_status": frame["boundary_status"].to_numpy(dtype=object),
        "z_hard_phat0": hard_soft["z_hard_phat0"].to_numpy(dtype=np.float64),
        "z_soft_barz0p5": hard_soft["z_soft_barz0p5"].to_numpy(dtype=np.float64),
    }
    return frame, metrics, arrays


def conditional_default_probability(
    model: torch.nn.Module,
    transition: CheckpointTransition,
    *,
    b_next: torch.Tensor,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Conditional default probability from the conditional productivity transition.

    Evaluates the hard default event ``Phat <= 0`` on the fixed bank children
    (each child carries the executed next-period leverage) and takes the
    child-weighted conditional expectation.  This is *not* ``bar_z``: the soft
    default probability ``1 - sigmoid(Phat/tau_z)`` is a different object and
    is never relabelled here.
    """
    device = _module_device(model)
    children = transition.children[..., :7].clone().to(device)
    children[..., 0:1] = b_next.reshape(-1, 1, 1).to(device=children.device, dtype=children.dtype)
    n_parent, n_child, state_dim = children.shape
    flat = children.reshape(-1, state_dim)
    phat_parts = []
    for start, stop in iter_row_chunks(flat.shape[0], chunk_size):
        phat_parts.append(model.forward_equity(flat[start:stop].to(device))["Phat"].detach())
    phat_child = torch.cat(phat_parts, dim=0).reshape(n_parent, n_child)
    default_child = (phat_child <= 0.0).to(torch.float64)
    weights = transition.branch_weights.to(device=device, dtype=torch.float64)
    return (weights * default_child).sum(dim=1).cpu().numpy()


def objective_slices(
    cp: EvalCheckpoint,
    transition: CheckpointTransition,
    *,
    config: PolicyEvalConfig,
    teacher=None,
    model: Optional[torch.nn.Module] = None,
    representative: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """P04 objective slices: Bellman RHS vs candidate bp + Q_claim overlay.

    For representative bank states the full candidate-bp curve of the Bellman
    RHS (cashflow + continuation) is returned together with the issuance
    ``Q_claim`` at every candidate.  On eta = 0 rows the objective must be flat
    in the candidate (the financing gate forces ``b_next = b_current``), which
    is checked and reported as ``eta0_objective_flatness_max_abs``.
    """
    model = model if model is not None else cp.models["policy_value"]
    teacher = teacher if teacher is not None else build_grid_teacher(
        cp,
        target_model=resolve_teacher_target_model(cp, teacher_target=config.teacher_target),
        max_expanded_states=config.teacher_max_expanded_states,
    )
    device = _module_device(model)
    parent = transition.parent_states.to(device)
    children_list, m_used, _m_raw, weights = split_transition(transition, device=device)
    n_rows = int(parent.shape[0])
    if representative is None:
        count = max(1, int(config.representative_count))
        representative = sorted(
            {int(round(index)) for index in np.linspace(0, n_rows - 1, count)}
        )
    candidates = torch.linspace(
        float(teacher.grid_min),
        float(teacher.grid_max),
        steps=int(config.objective_slice_points),
        dtype=parent.dtype,
        device=parent.device,
    ).reshape(1, -1)

    records: List[Dict[str, Any]] = []
    flatness: List[float] = []
    for row_index in representative:
        row = int(row_index)
        parent_row = parent[row:row + 1]
        children_row = [item[row:row + 1] for item in children_list]
        m_row = m_used[row:row + 1]
        weights_row = weights[row:row + 1]
        grid_row = candidates.expand(1, candidates.shape[1])
        q_issue = q_issue_claim_at_candidates(model, parent_row, grid_row)
        eta_value = float(parent_row[0, 2])
        for branch in ("p0", "pi"):
            values = _evaluate_grid_chunked(
                teacher,
                parent_row,
                children_row,
                m_row,
                grid_row,
                branch=branch,
                child_weights=weights_row,
            )
            value_grid = values["value_grid"][0].detach().to(torch.float64).cpu().numpy()
            cashflow = values["cashflow_grid_mean"][0].detach().to(torch.float64).cpu().numpy()
            continuation = values["continuation_grid_mean"][0].detach().to(torch.float64).cpu().numpy()
            p_child = values["p_child_grid_mean"][0].detach().to(torch.float64).cpu().numpy()
            default_share = values["default_grid_mean"][0].detach().to(torch.float64).cpu().numpy()
            q_claim = q_issue[0].detach().to(torch.float64).cpu().numpy()
            for index, bp_value in enumerate(grid_row[0].detach().to(torch.float64).cpu().numpy()):
                records.append(
                    {
                        "row_index": row,
                        "state_label": f"row_{row}",
                        "branch": branch,
                        "eta": eta_value,
                        "refi_active": 1.0 if eta_value > 0.5 else 0.0,
                        "bp_candidate": float(bp_value),
                        "bellman_rhs": float(value_grid[index]),
                        "cashflow": float(cashflow[index]),
                        "continuation": float(continuation[index]),
                        "q_claim_issue": float(q_claim[index]),
                        "p_child_mean": float(p_child[index]),
                        "default_share_mean": float(default_share[index]),
                    }
                )
            if eta_value <= 0.5:
                flatness.append(float(np.nanmax(value_grid) - np.nanmin(value_grid)))
    metrics = {
        "objective_slice_states": float(len(list(representative))),
        "objective_slice_eta0_flatness_max_abs": (
            float(max(flatness)) if flatness else float("nan")
        ),
    }
    return pd.DataFrame(records), metrics


def evaluate_firm_policy(
    cp: EvalCheckpoint,
    bank: Mapping[str, Any],
    *,
    config: Optional[PolicyEvalConfig] = None,
) -> PolicyEvalResult:
    """Evaluate the value/policy surfaces and boundaries of ``cp`` on a bank entry."""
    config = config if config is not None else PolicyEvalConfig()
    model = cp.models["policy_value"]
    device = _module_device(model)
    n_rows = int(torch.as_tensor(bank["parent_states"]).shape[0])
    labels = source_labels(bank, n_rows)
    metrics: Dict[str, float] = {}
    tables: Dict[str, pd.DataFrame] = {}
    surfaces: Dict[str, np.ndarray] = {}

    with torch.inference_mode():
        for module in cp.models.values():
            module.eval()
        transition = build_transition_for_checkpoint(
            _bank_payload_on_device(bank, device),
            cp.models["sdf_fc1"],
            cp.hyperparams,
            cp.economic_config,
        )
        teacher = build_grid_teacher(
            cp,
            target_model=resolve_teacher_target_model(cp, teacher_target=config.teacher_target),
            max_expanded_states=config.teacher_max_expanded_states,
        )
        policy_rows, policy_diag = grid_policy_solution(
            cp,
            transition,
            config=config,
            model=model,
            teacher=teacher,
            source=labels,
        )
        slices, slice_metrics = objective_slices(
            cp, transition, config=config, teacher=teacher, model=model
        )

    metrics.update(slice_metrics)

    # ---- financing-gate identity and Grid-BP vs head diagnostics ----
    eta = policy_rows["eta"].to_numpy(dtype=np.float64)
    eta0 = eta <= 0.5
    eta1 = ~eta0
    b_values = policy_rows["b"].to_numpy(dtype=np.float64)
    for branch, star_col, next_col in (
        ("p0", "bp0_star", "b_next_p0"),
        ("pi", "bpi_star", "b_next_pi"),
        ("mix", "bp_star_mix", "b_next_mix"),
    ):
        star = policy_rows[star_col].to_numpy(dtype=np.float64)
        nxt = policy_rows[next_col].to_numpy(dtype=np.float64)
        if eta0.any():
            metrics[f"eta0_financing_gate_max_error_{branch}"] = float(
                np.max(np.abs(nxt[eta0] - b_values[eta0]))
            )
        else:
            metrics[f"eta0_financing_gate_max_error_{branch}"] = float("nan")
        if eta1.any():
            metrics[f"eta1_financing_gate_max_error_{branch}"] = float(
                np.max(np.abs(nxt[eta1] - star[eta1]))
            )
        else:
            metrics[f"eta1_financing_gate_max_error_{branch}"] = float("nan")

    metrics["policy_rows_n"] = float(n_rows)
    metrics["policy_rows_eta0_n"] = float(eta0.sum())
    metrics["policy_rows_eta1_n"] = float(eta1.sum())

    def _eta1_stats(values: np.ndarray, prefix: str) -> None:
        subset = values[eta1]
        finite = subset[np.isfinite(subset)]
        metrics[f"{prefix}_n"] = float(subset.size)
        if finite.size == 0:
            for key in ("mean", "p90", "max"):
                metrics[f"{prefix}_{key}"] = float("nan")
        else:
            metrics[f"{prefix}_mean"] = float(finite.mean())
            metrics[f"{prefix}_p90"] = float(np.quantile(finite, 0.90))
            metrics[f"{prefix}_max"] = float(finite.max())

    _eta1_stats(policy_rows["bp0_star"].to_numpy(), "bp0_star_eta1")
    _eta1_stats(policy_rows["bpi_star"].to_numpy(), "bpi_star_eta1")
    _eta1_stats(policy_rows["bp_star_mix"].to_numpy(), "bp_star_mix_eta1")
    _eta1_stats(policy_rows["margin_p0"].to_numpy(), "grid_margin_p0_eta1")
    _eta1_stats(policy_rows["margin_pi"].to_numpy(), "grid_margin_pi_eta1")
    _eta1_stats(policy_rows["confidence_p0"].to_numpy(), "grid_confidence_p0_eta1")

    head_gap_p0 = np.abs(
        policy_rows["bp0_head"].to_numpy() - policy_rows["bp0_star"].to_numpy()
    )
    head_gap_pi = np.abs(
        policy_rows["bpI_head"].to_numpy() - policy_rows["bpi_star"].to_numpy()
    )
    _eta1_stats(head_gap_p0, "bp_head_vs_grid_p0_abs_eta1")
    _eta1_stats(head_gap_pi, "bp_head_vs_grid_pi_abs_eta1")
    if eta1.any():
        metrics["bp_head_match_rate_p0_eta1"] = float(
            (head_gap_p0[eta1] <= float(config.bp_match_tol)).mean()
        )
        metrics["bp_head_match_rate_pi_eta1"] = float(
            (head_gap_pi[eta1] <= float(config.bp_match_tol)).mean()
        )
    else:
        metrics["bp_head_match_rate_p0_eta1"] = float("nan")
        metrics["bp_head_match_rate_pi_eta1"] = float("nan")
    _eta1_stats(policy_rows["regret_p0_head"].to_numpy(), "bp_head_regret_p0_eta1")
    _eta1_stats(policy_rows["regret_pi_head"].to_numpy(), "bp_head_regret_pi_eta1")
    _eta1_stats(
        np.abs(policy_rows["bp0_star"].to_numpy() - policy_rows["bp_star_grid_p0"].to_numpy()),
        "bp_star_vs_grid_argmax_p0_eta1",
    )
    metrics["grid_boundary_low_share_p0_eta1"] = (
        float(policy_rows.loc[eta1, "boundary_low_p0"].mean()) if eta1.any() else float("nan")
    )
    metrics["grid_boundary_high_share_p0_eta1"] = (
        float(policy_rows.loc[eta1, "boundary_high_p0"].mean()) if eta1.any() else float("nan")
    )

    # ---- Bar_i_model vs Bar_i_executed (panel semantics) ----
    bar_i_model = policy_rows["bar_i_model"].to_numpy(dtype=np.float64)
    bar_i_executed = policy_rows["bar_i_executed"].to_numpy(dtype=np.float64)
    metrics["bar_i_model_mean"] = float(np.nanmean(bar_i_model))
    metrics["bar_i_executed_mean"] = float(np.nanmean(bar_i_executed))
    metrics["bar_i_model_vs_executed_abs_mean"] = float(
        np.nanmean(np.abs(bar_i_model - bar_i_executed))
    )
    metrics["investment_executed_share_rows"] = float(np.nanmean(bar_i_executed > 0.5))
    metrics["investment_model_share_rows"] = float(np.nanmean(bar_i_model > 0.5))

    # ---- conditional default probability (not bar_z) ----
    b_next_mix = torch.as_tensor(
        policy_rows["b_next_mix"].to_numpy(), dtype=transition.parent_states.dtype, device=transition.parent_states.device
    ).reshape(-1, 1)
    conditional_default = conditional_default_probability(
        model, transition, b_next=b_next_mix, chunk_size=config.chunk_size
    )
    bar_z_rows = policy_rows["bar_z"].to_numpy(dtype=np.float64)
    metrics["conditional_default_prob_mean"] = float(np.nanmean(conditional_default))
    metrics["conditional_default_prob_p90"] = float(
        np.nanquantile(conditional_default, 0.9)
    )
    metrics["bar_z_mean_rows"] = float(np.nanmean(bar_z_rows))
    metrics["hard_default_share_rows"] = float(
        np.nanmean(policy_rows["Phat"].to_numpy() <= 0.0)
    )
    metrics["conditional_default_vs_bar_z_abs_mean"] = float(
        np.nanmean(np.abs(conditional_default - bar_z_rows))
    )
    policy_rows["conditional_default_prob"] = conditional_default

    # ---- per-eta-view surfaces, investment and default boundaries ----
    investment_frames: List[pd.DataFrame] = []
    default_frames: List[pd.DataFrame] = []
    hard_soft_frames: List[pd.DataFrame] = []
    for view in config.eta_views:
        prefix = f"eta{int(round(float(view)))}_"
        reference = build_policy_reference(bank, eta=float(view), config=config)
        grid = build_policy_grid(bank, reference, config=config, device=device)
        view_surfaces = evaluate_firm_surfaces(
            model, grid, reference, chunk_size=int(config.chunk_size)
        )
        for name, values in view_surfaces.items():
            surfaces[f"{prefix}{name}"] = np.asarray(values)

        investment_frame, investment_metrics, investment_arrays = solve_investment_boundary(
            model,
            grid,
            reference,
            config=config,
            survival_mask=np.asarray(view_surfaces["survival_mask"], dtype=bool),
            i_max=(
                float(config.i_max)
                if config.i_max is not None
                else float(cp.economic_config.I_THRESHOLD)
            ),
        )
        default_frame, default_metrics, default_arrays = solve_default_boundary(
            grid,
            np.asarray(view_surfaces["Phat"], dtype=np.float64),
            np.asarray(view_surfaces["bar_z"], dtype=np.float64),
        )
        margin_arrays, margin_metrics = investment_margin_diagnostics(
            view_surfaces,
            {
                "survival_mask": np.asarray(view_surfaces["survival_mask"], dtype=bool),
                "investment_status": investment_arrays["investment_status_raw"],
            },
        )
        for key, value in investment_metrics.items():
            metrics[f"{prefix}{key}"] = value
        for key, value in default_metrics.items():
            metrics[f"{prefix}{key}"] = value
        for key, value in margin_metrics.items():
            metrics[f"{prefix}{key}"] = value
        for name, values in investment_arrays.items():
            surfaces[f"{prefix}investment_{name}"] = np.asarray(values)
        for name, values in default_arrays.items():
            surfaces[f"{prefix}default_{name}"] = np.asarray(values)
        for name, values in margin_arrays.items():
            surfaces[f"{prefix}{name}"] = np.asarray(values)

        investment_frame = investment_frame.copy()
        investment_frame["eta_view"] = float(view)
        default_frame = default_frame.copy()
        default_frame["eta_view"] = float(view)
        hard_soft = default_frame.copy()
        hard_soft["z_hard_phat0"] = default_arrays["z_hard_phat0"]
        hard_soft["z_soft_barz0p5"] = default_arrays["z_soft_barz0p5"]
        investment_frames.append(investment_frame)
        default_frames.append(default_frame)
        hard_soft_frames.append(hard_soft)

    tables["policy_rows"] = policy_rows
    tables["investment_boundary"] = pd.concat(investment_frames, ignore_index=True)
    tables["default_boundary"] = pd.concat(default_frames, ignore_index=True)
    tables["hard_soft_boundary"] = pd.concat(hard_soft_frames, ignore_index=True)
    tables["objective_slices"] = slices

    record = semantic_record(cp)
    diagnostics: Dict[str, Any] = {
        "formal_policy": "grid_bp (training.bp_grid_teacher.BPGridTeacher)",
        "bp_head_semantics": (
            "bp0/bpI/bp head outputs are compatibility columns only; the formal "
            "policy is the Grid-BP greedy solution and b_next is computed from it"
        ),
        "financing_gate": "b_next = eta_t * bp + (1 - eta_t) * b_t (parent eta_t)",
        "bar_z_semantics": (
            "bar_z = 1 - sigmoid(Phat/tau_z) is a soft default probability, not a "
            "productivity cutoff; conditional_default_prob is computed from the "
            "conditional productivity transition on the fixed bank shocks"
        ),
        "bar_i_semantics": (
            "bar_i_model = model investment probability (bar_i_cond); "
            "bar_i_executed = executed action (bar_i_eff; forced 0 at entry/birth "
            "nodes in the panel)"
        ),
        "investment_boundary": (
            "zeros of PI - V0 along i; no-uniqueness is reported "
            "(always_invest / never_invest / single_crossing / multiple_crossings)"
        ),
        "default_boundary": "zeros of Phat(b, z) = 0 along z (extract_phat_default_boundary)",
        "teacher_grid": policy_diag.get("teacher_grid", {}),
        "teacher_target": config.teacher_target,
        "q_parameterization": record.get("q_parameterization"),
        "value_parameterization": dict(record.get("value_parameterization") or {}),
        "transition_metadata": dict(transition.metadata),
        "bank_id": bank.get("bank_id"),
        "bank_tags": list(bank.get("tags") or []),
        "episode": cp.episode,
        "parent_states_sha256": content_sha256(transition.parent_states.detach().cpu()),
        "policy_flip_tol": float(config.policy_flip_tol),
        "bp_match_tol": float(config.bp_match_tol),
        "eta_views": [float(view) for view in config.eta_views],
        "evaluation_only": True,
        "optimizer_steps": 0,
        "model_mode": "eval",
    }
    return PolicyEvalResult(
        metrics=metrics,
        tables=tables,
        surfaces=surfaces,
        diagnostics=diagnostics,
    )


def compare_policy_across_episodes(
    results: Sequence[PolicyEvalResult],
) -> pd.DataFrame:
    """Cross-episode policy drift on the fixed bank states (Phase D input).

    Every result must come from the *same* fixed bank rows (checked through the
    recorded ``parent_states_sha256``); otherwise the comparison is refused.
    For consecutive episodes the table reports:

    * ``bp_star_drift_*`` -- ``|bp*_e - bp*_{e-1}|`` on eta = 1 rows
      (``bp_star_mix``; branch-wise drifts are reported alongside);
    * ``policy_flip_rate`` -- share of eta = 1 rows whose mixture leverage moves
      by more than ``policy_flip_tol``;
    * ``default_flip_rate`` -- share of rows whose hard default indicator
      (``Phat <= 0``) flips;
    * ``investment_flip_rate`` -- share of rows whose hard investment decision
      (``PI > P0`` at the bank row's own investment cost) flips.
    """
    results = list(results)
    if len(results) < 2:
        raise ValueError("compare_policy_across_episodes requires at least two results")
    fingerprints = {
        str(result.diagnostics.get("parent_states_sha256")) for result in results
    }
    if len(fingerprints) != 1:
        raise ValueError(
            "policy results were not computed on the same fixed bank states "
            f"(parent_states_sha256 differs: {sorted(fingerprints)})"
        )

    rows: List[Dict[str, Any]] = []
    for previous, current in zip(results[:-1], results[1:]):
        prev = previous.tables["policy_rows"]
        cur = current.tables["policy_rows"]
        if len(prev) != len(cur):
            raise ValueError("policy_rows length differs across episodes")
        if not np.array_equal(
            prev["row_index"].to_numpy(), cur["row_index"].to_numpy()
        ):
            raise ValueError("policy_rows row_index alignment differs across episodes")
        tol = float(
            current.diagnostics.get(
                "policy_flip_tol", previous.diagnostics.get("policy_flip_tol", 1e-2)
            )
        )
        eta = cur["eta"].to_numpy(dtype=np.float64)
        eta1 = eta > 0.5
        drift_mix = np.abs(
            cur["bp_star_mix"].to_numpy() - prev["bp_star_mix"].to_numpy()
        )
        drift_p0 = np.abs(cur["bp0_star"].to_numpy() - prev["bp0_star"].to_numpy())
        drift_pi = np.abs(cur["bpi_star"].to_numpy() - prev["bpi_star"].to_numpy())
        drift_active = drift_mix[eta1]

        default_prev = prev["Phat"].to_numpy() <= 0.0
        default_cur = cur["Phat"].to_numpy() <= 0.0
        invest_prev = prev["PI"].to_numpy() > prev["P0"].to_numpy()
        invest_cur = cur["PI"].to_numpy() > cur["P0"].to_numpy()

        def _stat(values: np.ndarray, prefix: str) -> Dict[str, float]:
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return {
                    f"{prefix}_mean": float("nan"),
                    f"{prefix}_p90": float("nan"),
                    f"{prefix}_max": float("nan"),
                }
            return {
                f"{prefix}_mean": float(finite.mean()),
                f"{prefix}_p90": float(np.quantile(finite, 0.90)),
                f"{prefix}_max": float(finite.max()),
            }

        record: Dict[str, Any] = {
            "episode_previous": previous.diagnostics.get("episode"),
            "episode_current": current.diagnostics.get("episode"),
            "n_rows": int(len(cur)),
            "n_eta1_rows": int(eta1.sum()),
            "policy_flip_tol": tol,
            "policy_flip_rate": (
                float((drift_active > tol).mean()) if drift_active.size else float("nan")
            ),
            "default_flip_rate": float((default_prev != default_cur).mean()),
            "investment_flip_rate": float((invest_prev != invest_cur).mean()),
        }
        record.update(_stat(drift_active, "bp_star_drift"))
        record.update(_stat(drift_p0[eta1], "bp0_star_drift"))
        record.update(_stat(drift_pi[eta1], "bpi_star_drift"))
        rows.append(record)
    return pd.DataFrame(rows)
