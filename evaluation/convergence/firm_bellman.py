"""Firm Bellman / value evaluator for the equilibrium convergence evaluator.

This module computes the firm-side Bellman evidence of one checkpoint on one
fixed evaluation bank entry (``evaluation/convergence/eval_bank.py``).  Children
are always rebuilt from the bank's fixed shocks through *this checkpoint's*
FC1/SDF via :func:`build_transition_for_checkpoint`; no simulation and no
training episode is ever instantiated.

Caliber discipline (the economic red lines of the protocol):

* **Grid BP is the formal policy**.  The bp0/bpI/bp heads of
  ``PolicyValueModel`` are compatibility outputs only and are never used as the
  formal policy; the greedy leverage used by every residual here is solved by
  ``training.bp_grid_teacher.BPGridTeacher``.
* **Financing gate**: ``b_next = eta_t * bp + (1 - eta_t) * b_t`` with the
  *parent* eta.  eta = 0 rows are forced (``b_next = b_current``), so every BP
  statistic is restricted to refinancing-active (eta = 1) rows.
* **Q four calibers** are kept separate: ``q_unit`` (unit bond price),
  ``Q_claim = b_+ * q_unit`` (issuance pricing / Bellman), ``Q_effective``
  (settlement; only ``hybrid_regime`` has the realized-default hard gate) and
  ``recovery = phi * (1 - delta + exp(x + z))`` (``asset_only``).  Issuance and
  Bellman residuals only ever use the claim caliber.
* **Physical vs Normalized P0/PI**: ``exp_xz`` value parameterization makes the
  head outputs *normalized*; ``physical = equity_value_scale * normalized`` and
  the normalized residual is ``R / (1 + exp(clamp(x + z)))``.  Both calibers are
  always reported and annotated.
* **Frozen-policy vs fresh-greedy residuals** are two different metrics.  The
  frozen residual freezes the greedy policy / selector / survival / Q at build
  time; the fresh-greedy residual re-solves the greedy RHS (``value_star``) and
  measures ``P - value_star``.  Optimality evidence prefers the fresh one.

Everything runs ``model.eval()`` + ``torch.inference_mode()``; no optimizer is
ever constructed (``checkpoint_io.count_optimizer_steps()`` is 0).

Frozen-policy residual math is **reused**, not rewritten: the frozen batches are
rebuilt from the bank transition into ``FrozenPolicyBatch`` structures and
processed by ``training.frozen_policy_p_solver`` (``build_frozen_branch`` /
``residual_tensors`` / ``evaluate_problem_rows``), so the evaluator reproduces
the training-side residual tensors exactly.  The Episode dependency of
``build_frozen_policy_batch`` (``_build_policy_m_lists`` /
``_expand_policy_expectation_children``) is bypassed because
``build_transition_for_checkpoint`` already delivers exact-eta expanded children
with normalized child weights and clipped/raw M -- the same structures the
Episode helpers produce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from analysis.economic_config import AnalysisEconomicConfig
from losses import P0Loss, PILoss
from losses.q_loss import (
    compute_parent_default_regime_weights,
    compute_q_survival_recovery_components,
    compute_recovery_target,
    resolve_q_parent_default_regime_mode,
    resolve_recovery_normalization_mode,
)
from training.bp_grid_teacher import BPGridTeacher, _slice_child_axis, resolve_grid_chunk_plan
from training.frozen_policy_p_solver import (
    FrozenPolicyBatch,
    build_frozen_branch,
    evaluate_problem_rows,
)

from .checkpoint_io import EvalCheckpoint, semantic_record
from .eval_bank import CheckpointTransition, build_transition_for_checkpoint, content_sha256


# Source labels: 0=Simulated / 1=Coverage / -1=unknown.  Bank C is synthetic
# stress, which is distinguished through the bank tags by ``source_labels``.
SOURCE_LABEL_BY_ID = {0: "simulated", 1: "coverage"}
SOURCE_LABELS = ("simulated", "coverage", "stress", "unknown")

FRESH_GREEDY_SOURCES = ("reuse_build", "resolve")

# Residual caliber notes recorded in every result (never silently mixed).
CALIBER_NOTE = (
    "physical = head/output values in physical units; normalized = R / "
    "equity_value_scale where equity_value_scale = 1 + exp(clamp(x+z, max=log_max)) "
    "for value_scale_mode='exp_xz' and 1 otherwise (not a z-score). "
    "trainM uses pv-clipped M (pv_use_clipped_m), rawM uses the raw SDF M."
)


def _module_device(model: torch.nn.Module) -> torch.device:
    """Execution device of a model: the device of its parameters.

    The model parameter device is authoritative: every chunk tensor must be
    moved onto it before a forward call, and every statistic is moved back with
    ``.cpu()`` before numpy export.
    """
    for parameter in model.parameters():
        return parameter.device
    return torch.device("cpu")


def _bank_payload_on_device(
    bank: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Shallow-copy of a bank payload with its tensors moved to ``device``.

    ``build_transition_for_checkpoint`` has no device parameter and rebuilds the
    transition on the device of ``parent_states``; moving the payload here (in
    the evaluator, without touching ``eval_bank``) lands every transition tensor
    on the model device before any forward runs.
    """
    moved: Dict[str, Any] = dict(bank)
    for key in ("parent_states", "hatc_cal", "lnk_cal"):
        value = bank.get(key)
        if torch.is_tensor(value):
            moved[key] = value.to(device)
    shock_payload = bank.get("shock_bank")
    if isinstance(shock_payload, Mapping):
        moved["shock_bank"] = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in shock_payload.items()
        }
    return moved


def build_bellman_losses(
    economic_config: AnalysisEconomicConfig,
) -> Tuple[P0Loss, PILoss]:
    """P0/PI cashflow functions parameterised by the checkpoint's economics.

    Mirrors ``evaluation.bellman_diagnostics.evaluate_bellman_residuals`` so the
    evaluator never picks up the current ``Config`` by accident.
    """
    p0_loss = P0Loss(
        delta=economic_config.DELTA,
        tau=economic_config.TAU,
        kappa_b=economic_config.KAPPA_B,
        kappa_e=economic_config.KAPPA_E,
        aio_weight=economic_config.AIO_WEIGHT,
        alpha_z=economic_config.ALPHA_Z,
        beta_z=economic_config.BETA_Z,
        z0=economic_config.Z0,
    )
    pi_loss = PILoss(
        delta=economic_config.DELTA,
        tau=economic_config.TAU,
        g=economic_config.G,
        kappa_b=economic_config.KAPPA_B,
        kappa_e=economic_config.KAPPA_E,
        aio_weight=economic_config.AIO_WEIGHT,
        alpha_z=economic_config.ALPHA_Z,
        beta_z=economic_config.BETA_Z,
        z0=economic_config.Z0,
        b_penalty_weight=0.0,
    )
    return p0_loss, pi_loss


def resolve_teacher_target_model(
    cp: EvalCheckpoint,
    *,
    teacher_target: str = "policy_value",
) -> torch.nn.Module:
    """Model whose P/Q drive the Grid-BP teacher.

    ``policy_value`` (default) evaluates the checkpoint self-consistently: the
    greedy policy is re-solved from the evaluated model's own P/Q, which is what
    the fresh-greedy optimality metric means.  ``firm_target`` reproduces the
    training-side teacher when a frozen target network was stored in the
    checkpoint (``Episode._target_policy_value`` semantics).
    """
    choice = str(teacher_target).strip().lower()
    if choice == "firm_target":
        target = cp.models.get("firm_target")
        if target is None:
            raise ValueError(
                "teacher_target='firm_target' requires models['firm_target'] in the "
                "checkpoint; this checkpoint does not carry one"
            )
        return target
    if choice != "policy_value":
        raise ValueError(
            f"teacher_target must be 'policy_value' or 'firm_target', got {teacher_target!r}"
        )
    return cp.models["policy_value"]


def build_grid_teacher(
    cp: EvalCheckpoint,
    *,
    target_model: Optional[torch.nn.Module] = None,
    q_target_model: Optional[torch.nn.Module] = None,
    max_expanded_states: Optional[int] = None,
) -> BPGridTeacher:
    """Build the production Grid-BP teacher from the checkpoint's HyperParams."""
    target = target_model if target_model is not None else cp.models["policy_value"]
    q_target = q_target_model if q_target_model is not None else target
    p0_loss, pi_loss = build_bellman_losses(cp.economic_config)
    return BPGridTeacher.from_hyperparams(
        target,
        p0_loss,
        pi_loss,
        cp.hyperparams,
        q_target_model=q_target,
        max_expanded_states_override=max_expanded_states,
    )


def split_transition(
    transition: CheckpointTransition,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a checkpoint transition into teacher/frozen-solver inputs.

    Returns ``(children_list, m_used, m_raw, child_weights)`` where the child
    lists are ``2J`` tensors of shape ``[N, 7]`` (exact-eta expanded) and the M
    tensors keep the ``[N, 2J, 1]`` layout required by ``BPGridTeacher``.
    ``device`` moves every returned tensor onto the model device before any
    forward (a no-op when the transition already lives there).
    """
    children = transition.children
    n_parent, n_child, _ = children.shape
    children_list = [children[:, index, :] for index in range(int(n_child))]
    m_used = transition.m_used.reshape(n_parent, n_child, 1)
    m_raw = transition.m_raw.reshape(n_parent, n_child, 1)
    weights = transition.branch_weights.reshape(n_parent, n_child)
    if device is not None:
        children_list = [item.to(device) for item in children_list]
        m_used = m_used.to(device)
        m_raw = m_raw.to(device)
        weights = weights.to(device)
    return children_list, m_used, m_raw, weights


def iter_row_chunks(n_rows: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
    step = max(1, int(chunk_size))
    for start in range(0, int(n_rows), step):
        yield start, min(start + step, int(n_rows))


def source_labels(bank: Mapping[str, Any], n_rows: int) -> List[str]:
    """Per-row source label: simulated / coverage / stress / unknown."""
    stress = "out-of-distribution-stress" in list(bank.get("tags") or [])
    ids = bank.get("source_id")
    labels: List[str] = []
    if ids is not None and len(ids) == n_rows:
        for value in ids:
            key = int(value)
            label = SOURCE_LABEL_BY_ID.get(key)
            if label is None:
                label = "stress" if stress else "unknown"
            labels.append(label)
    else:
        labels = ["stress" if stress else "unknown"] * int(n_rows)
    return labels


def _quantile(values: torch.Tensor, q: float) -> float:
    flat = values.detach().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    return float(torch.quantile(flat.to(torch.float64), float(q)))


def residual_stat_block(
    values: Any,
    *,
    prefix: str,
    level: Optional[Any] = None,
    eta: Optional[Any] = None,
    source: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Full residual statistics for one residual caliber.

    Reports MAE / RMSE / P90 / P99 / Max / signed mean / sample counts /
    nonfinite ratio, the relative-scale error ``E[|d|]/E[1+|level|]``, and eta /
    source strata.  All values are finite-only diagnostics; the nonfinite ratio
    is reported separately so a collapsed region can never look accurate.
    """
    tensor = torch.as_tensor(values).detach().reshape(-1).to(torch.float64)
    finite = torch.isfinite(tensor)
    out: Dict[str, float] = {
        f"{prefix}_n": float(tensor.numel()),
        f"{prefix}_n_finite": float(finite.sum()),
        f"{prefix}_nonfinite_ratio": (
            float((~finite).sum()) / float(tensor.numel()) if tensor.numel() else 0.0
        ),
    }
    good = tensor[finite]
    if good.numel() == 0:
        for key in ("mae", "rmse", "p90", "p99", "max", "mean_signed"):
            out[f"{prefix}_{key}"] = float("nan")
    else:
        absolute = good.abs()
        out[f"{prefix}_mae"] = float(absolute.mean())
        out[f"{prefix}_rmse"] = float(good.square().mean().sqrt())
        out[f"{prefix}_p90"] = _quantile(absolute, 0.90)
        out[f"{prefix}_p99"] = _quantile(absolute, 0.99)
        out[f"{prefix}_max"] = float(absolute.max())
        out[f"{prefix}_mean_signed"] = float(good.mean())
    if level is not None:
        level_tensor = torch.as_tensor(level).detach().reshape(-1).to(torch.float64)
        if level_tensor.numel() == tensor.numel():
            pair_finite = finite & torch.isfinite(level_tensor)
            if int(pair_finite.sum()) > 0:
                numerator = float(tensor[pair_finite].abs().mean())
                denominator = float((1.0 + level_tensor[pair_finite].abs()).mean())
                out[f"{prefix}_relative_mae"] = numerator / max(denominator, 1e-12)
            else:
                out[f"{prefix}_relative_mae"] = float("nan")
        else:
            out[f"{prefix}_relative_mae"] = float("nan")
    if eta is not None:
        eta_tensor = torch.as_tensor(eta).detach().reshape(-1).to(torch.float64)
        if eta_tensor.numel() == tensor.numel():
            for label, mask in (("eta0", eta_tensor <= 0.0), ("eta1", eta_tensor > 0.0)):
                subset = tensor[mask]
                subset_finite = subset[torch.isfinite(subset)]
                out[f"{prefix}_{label}_n"] = float(subset.numel())
                out[f"{prefix}_{label}_n_finite"] = float(subset_finite.numel())
                out[f"{prefix}_{label}_nonfinite_ratio"] = (
                    float(subset.numel() - subset_finite.numel()) / float(subset.numel())
                    if subset.numel()
                    else 0.0
                )
                if subset_finite.numel() == 0:
                    out[f"{prefix}_{label}_mae"] = float("nan")
                    out[f"{prefix}_{label}_p90"] = float("nan")
                else:
                    out[f"{prefix}_{label}_mae"] = float(subset_finite.abs().mean())
                    out[f"{prefix}_{label}_p90"] = _quantile(subset_finite.abs(), 0.90)
    if source is not None and len(source) == tensor.numel():
        for label in SOURCE_LABELS:
            mask = torch.tensor([str(item) == label for item in source], dtype=torch.bool)
            subset = tensor[mask]
            subset_finite = subset[torch.isfinite(subset)]
            out[f"{prefix}_{label}_n"] = float(subset.numel())
            if subset_finite.numel() == 0:
                out[f"{prefix}_{label}_mae"] = float("nan")
                out[f"{prefix}_{label}_p90"] = float("nan")
            else:
                out[f"{prefix}_{label}_mae"] = float(subset_finite.abs().mean())
                out[f"{prefix}_{label}_p90"] = _quantile(subset_finite.abs(), 0.90)
    return out


def _forward_fields(
    model: torch.nn.Module,
    states: torch.Tensor,
    fields: Sequence[str],
    *,
    chunk_size: int,
) -> Dict[str, torch.Tensor]:
    device = _module_device(model)
    collected: Dict[str, List[torch.Tensor]] = {name: [] for name in fields}
    for start, stop in iter_row_chunks(states.shape[0], chunk_size):
        output = model(states[start:stop].to(device))
        for name in fields:
            if isinstance(output, dict):
                value = output[name]
            else:
                value = getattr(output, name)
            collected[name].append(value.detach())
    return {name: torch.cat(parts, dim=0) for name, parts in collected.items()}


def _call_chunked(fn, states: torch.Tensor, *, chunk_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    parts = []
    for start, stop in iter_row_chunks(states.shape[0], chunk_size):
        chunk = states[start:stop]
        if device is not None:
            chunk = chunk.to(device)
        parts.append(fn(chunk).detach())
    return torch.cat(parts, dim=0)


def q_caliber_tensors(
    model: torch.nn.Module,
    states: torch.Tensor,
    *,
    economic_config: AnalysisEconomicConfig,
    recovery_normalization_mode: str = "asset_only",
    chunk_size: int = 8192,
) -> Dict[str, torch.Tensor]:
    """Evaluate the four Q calibers plus the equity envelope on parent states.

    Returns ``q_unit`` / ``Q_claim`` / ``Q_effective`` / ``recovery`` (model
    side, ``asset_only`` formula) / ``recovery_target`` (mode-resolved target)
    together with ``Phat`` / ``P`` / ``bar_z`` / ``survival_prob``.

    ``Q_claim`` is the only caliber allowed in issuance pricing and Bellman
    residuals; ``Q_effective`` is the settlement caliber and only diverges from
    the claim in ``hybrid_regime`` (realized-default hard gate).
    """
    device = _module_device(model)
    states = states.to(device)
    equity = {
        name: _call_chunked(
            lambda chunk, field=name: model.forward_equity(chunk)[field],
            states,
            chunk_size=chunk_size,
            device=device,
        )
        for name in ("Phat", "P", "bar_z", "survival_prob")
    }
    phat = equity["Phat"]
    b = states[:, 0:1]
    x = states[:, 4:5]
    z = states[:, 1:2]
    q_parameterization = str(getattr(model, "q_parameterization", "direct"))
    q_claim = _call_chunked(model._q_claim_output, states, chunk_size=chunk_size, device=device)
    if q_parameterization in {"b_times_unit", "hybrid_regime"}:
        q_unit = _call_chunked(model._q_unit_output, states, chunk_size=chunk_size, device=device)
    else:
        # Direct mode predicts total debt value; q_unit is a derived reporting
        # ratio and is undefined at b <= 0 (no epsilon division).
        q_unit = torch.where(
            b > 0.0,
            q_claim / b.clamp_min(1e-12),
            torch.full_like(q_claim, float("nan")),
        )
    recovery_fn = getattr(model, "_q_recovery_output", None)
    if callable(recovery_fn):
        recovery = _call_chunked(recovery_fn, states, chunk_size=chunk_size, device=device)
    else:
        recovery = compute_recovery_target(
            b, x, z,
            phi=float(economic_config.PHI),
            delta=float(economic_config.DELTA),
            recovery_normalization_mode="asset_only",
        )
    recovery_target = compute_recovery_target(
        b, x, z,
        phi=float(economic_config.PHI),
        delta=float(economic_config.DELTA),
        recovery_normalization_mode=recovery_normalization_mode,
    )
    if q_parameterization == "hybrid_regime":
        effective_parts = []
        for start, stop in iter_row_chunks(states.shape[0], chunk_size):
            effective_parts.append(
                model._q_effective_output(
                    states[start:stop].to(device), phat=phat[start:stop].to(device)
                ).detach()
            )
        q_effective = torch.cat(effective_parts, dim=0)
    else:
        q_effective = q_claim
    return {
        "q_unit": q_unit,
        "Q_claim": q_claim,
        "Q_effective": q_effective,
        "recovery": recovery,
        "recovery_target": recovery_target,
        "Phat": phat,
        "P": equity["P"],
        "bar_z": equity["bar_z"],
        "survival_prob": equity["survival_prob"],
    }


def q_issue_claim_at_candidates(
    model: torch.nn.Module,
    parent_states: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """Issuance (new-debt) claim at candidate leverage values.

    Mirrors ``BPGridTeacher._q_issue_grid``'s pricing rule: issuance is always
    priced as a **live debt claim** at ``b := candidate``; the realized-default
    recovery substitution only exists in the settlement caliber and must never
    appear here.
    """
    if candidates.ndim != 2:
        raise ValueError("candidates must have shape [N, K]")
    device = _module_device(model)
    parent_states = parent_states.to(device)
    candidates = candidates.to(device)
    n_rows, n_grid = candidates.shape
    issue_state = (
        parent_states.unsqueeze(1)
        .expand(n_rows, n_grid, parent_states.shape[-1])
        .reshape(n_rows * n_grid, parent_states.shape[-1])
        .clone()
    )
    issue_state[:, 0:1] = candidates.reshape(-1, 1)
    q_parameterization = str(getattr(model, "q_parameterization", "direct"))
    if q_parameterization in {"b_times_unit", "hybrid_regime"}:
        claim = model._q_claim_output(issue_state)
    else:
        claim = model._q_output(issue_state)
    return claim.reshape(n_rows, n_grid)


def q_region_masks(
    b: np.ndarray,
    phat: np.ndarray,
    bar_z: np.ndarray,
    *,
    zero_b_eps: float = 0.0,
    phat_eps: float = 1e-2,
    bar_z_low: float = 0.1,
    bar_z_high: float = 0.9,
) -> Dict[str, np.ndarray]:
    """Q region masks (Prompt section 4 Q0/QD/QS/Q-polish as *state regions*).

    * ``q0`` -- zero-debt structural region (``b <= zero_b_eps``)
    * ``qd`` -- default / recovery region (``b > 0`` and ``Phat <= 0``)
    * ``qs`` -- survival region (``b > 0`` and ``Phat > 0``)
    * ``qboundary`` -- polish/boundary band (``|Phat| <= phat_eps`` or
      ``bar_z in [bar_z_low, bar_z_high]``); it *overlaps* qd/qs by design.

    ``bar_z`` is a soft default probability (``1 - sigmoid(Phat/tau_z)``), never
    a productivity cutoff; it only widens the boundary band here.
    """
    b_arr = np.asarray(b, dtype=np.float64).reshape(-1)
    phat_arr = np.asarray(phat, dtype=np.float64).reshape(-1)
    bar_z_arr = np.asarray(bar_z, dtype=np.float64).reshape(-1)
    if not (b_arr.shape == phat_arr.shape == bar_z_arr.shape):
        raise ValueError("b / Phat / bar_z must have the same number of rows")
    zero = b_arr <= float(zero_b_eps)
    default = (~zero) & (phat_arr <= 0.0)
    survival = (~zero) & (phat_arr > 0.0)
    finite_phat = np.isfinite(phat_arr)
    boundary = (
        (finite_phat & (np.abs(phat_arr) <= float(phat_eps)))
        | (
            np.isfinite(bar_z_arr)
            & (bar_z_arr >= float(bar_z_low))
            & (bar_z_arr <= float(bar_z_high))
        )
    )
    return {
        "q0": zero,
        "qd": default,
        "qs": survival,
        "qboundary": boundary,
    }


def q_bellman_targets(
    model: torch.nn.Module,
    transition: CheckpointTransition,
    *,
    economic_config: AnalysisEconomicConfig,
    q_calibers: Mapping[str, torch.Tensor],
    m_mode: str,
    recovery_normalization_mode: str = "asset_only",
    parent_default_regime_mode: str = "legacy_soft_penalty",
    parent_default_eps: Optional[float] = None,
    parent_default_tau: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Q Bellman / training targets for one M caliber (``trainM`` or ``rawM``).

    Reproduces ``evaluation.bellman_diagnostics.evaluate_bellman_residuals`` on
    bank rows: the claim recursion dilutes issued debt at
    ``b / (bar_i * (G - 1) + 1)`` and the target splits into survival and
    recovery components.  Only the **claim** caliber enters the recursion.
    """
    if m_mode not in {"trainM", "rawM"}:
        raise ValueError(f"m_mode must be 'trainM' or 'rawM', got {m_mode!r}")
    device = _module_device(model)
    parent = transition.parent_states.to(device)
    children = transition.children[..., :7].to(device)
    m_values = (transition.m_used if m_mode == "trainM" else transition.m_raw).to(device)
    weights = transition.branch_weights.unsqueeze(-1).to(device)

    q_claim = q_calibers["Q_claim"]
    b = parent[:, 0:1]
    bar_i = _forward_fields(model, parent, ("bar_i",), chunk_size=max(int(parent.shape[0]), 1))["bar_i"]
    phat = q_calibers["Phat"]

    multiplier = bar_i * (float(economic_config.G) - 1.0) + 1.0
    b_sp = b / multiplier.clamp_min(1e-6)
    q_child_states = children.clone()
    q_child_states[..., 0:1] = b_sp.unsqueeze(1)
    flat_q_child_states = q_child_states.reshape(-1, q_child_states.shape[-1])
    qsp = model._q_claim_output(flat_q_child_states).reshape(parent.shape[0], -1, 1)
    try:
        bar_zsp = _forward_fields(
            model, flat_q_child_states, ("bar_z",),
            chunk_size=max(int(flat_q_child_states.shape[0]), 1),
        )["bar_z"].reshape(parent.shape[0], -1, 1)
    except (AttributeError, KeyError):
        bar_zsp = torch.zeros_like(qsp)

    components = compute_q_survival_recovery_components(
        Q=q_claim.unsqueeze(1),
        b=b.unsqueeze(1),
        bar_i=bar_i.unsqueeze(1),
        M=m_values,
        Qsp=qsp,
        bar_z=bar_zsp,
        x_child=q_child_states[..., 4:5],
        z_child=q_child_states[..., 1:2],
        g=float(economic_config.G),
        delta=float(economic_config.DELTA),
        phi=float(economic_config.PHI),
        recovery_normalization_mode=recovery_normalization_mode,
    )
    q_target_survival = (weights * components["q_target_survival"]).sum(dim=1)
    q_target_recovery = (weights * components["q_target_recovery"]).sum(dim=1)
    q_target_bellman = q_target_survival + q_target_recovery

    regime = compute_parent_default_regime_weights(
        phat,
        mode=parent_default_regime_mode,
        eps=parent_default_eps,
        tau=parent_default_tau,
    )
    if str(getattr(model, "q_parameterization", "direct")) == "hybrid_regime":
        q_target_training = q_target_bellman
    else:
        q_target_training = (
            regime["parent_survival_weight"] * q_target_bellman
            + regime["parent_default_weight"] * q_calibers["recovery_target"]
        )
    return {
        "q_target_survival": q_target_survival,
        "q_target_recovery": q_target_recovery,
        "q_target_bellman": q_target_bellman,
        "q_target_training": q_target_training,
        "rq_bellman_signed": q_target_bellman - q_claim,
        "rq_training_signed": q_target_training - q_claim,
        "recovery_current": q_calibers["recovery_target"],
        "parent_survival_weight": regime["parent_survival_weight"],
        "parent_default_weight": regime["parent_default_weight"],
        "parent_hard_default": regime["parent_hard_default"],
        "bar_i": bar_i,
        "multiplier": multiplier,
    }


@dataclass
class BellmanEvalConfig:
    """Configuration of one firm Bellman evaluation run."""

    parent_chunk_size: int = 256
    chunk_size: int = 8192
    dense_grid_size: int = 81
    teacher_max_expanded_states: Optional[int] = None
    normalize_by_value_scale: Optional[bool] = None
    teacher_target: str = "policy_value"
    fresh_greedy_source: str = "reuse_build"
    q_recovery_normalization_mode: Optional[str] = None
    q_parent_default_regime_mode: Optional[str] = None
    q_parent_default_eps: Optional[float] = None
    q_parent_default_tau: Optional[float] = None
    zero_b_eps: float = 0.0
    q_boundary_phat_eps: float = 1e-2
    q_boundary_bar_z_low: float = 0.1
    q_boundary_bar_z_high: float = 0.9

    def __post_init__(self) -> None:
        if int(self.parent_chunk_size) < 1:
            raise ValueError("parent_chunk_size must be positive")
        if int(self.dense_grid_size) < 2:
            raise ValueError("dense_grid_size must be at least 2")
        if str(self.fresh_greedy_source) not in FRESH_GREEDY_SOURCES:
            raise ValueError(
                f"fresh_greedy_source must be one of {FRESH_GREEDY_SOURCES}, "
                f"got {self.fresh_greedy_source!r}"
            )


@dataclass
class BellmanEvalResult:
    """Result of :func:`evaluate_firm_bellman`.

    * ``metrics`` -- flat float metrics (blocks: ``frozen_*`` / ``fresh_*`` /
      ``q_<region>_*`` / ``parity_*`` / ``bp_grid_*`` / ``optimizer_steps``).
    * ``tables`` -- CSV-ready DataFrames (``rows``, ``residual_stats``,
      ``q_region_stats``, ``bp_discretization``, ``parity``).
    * ``surfaces`` -- per-row (or b x z reshaped) arrays for plotting.
    * ``diagnostics`` -- caliber notes, semantic record, transition metadata.
    """

    metrics: Dict[str, float]
    tables: Dict[str, pd.DataFrame]
    surfaces: Dict[str, np.ndarray]
    diagnostics: Dict[str, Any]


def _resolve_recovery_mode(cp: EvalCheckpoint, config: BellmanEvalConfig) -> Tuple[str, str]:
    record = semantic_record(cp)
    if config.q_recovery_normalization_mode is not None:
        return (
            resolve_recovery_normalization_mode(config.q_recovery_normalization_mode),
            "config_override",
        )
    return (
        resolve_recovery_normalization_mode(record["q_recovery_normalization_mode"]),
        record["q_recovery_normalization_mode_source"],
    )


def build_frozen_policy_problem(
    cp: EvalCheckpoint,
    transition: CheckpointTransition,
    *,
    config: BellmanEvalConfig,
    model: Optional[torch.nn.Module] = None,
    q_model: Optional[torch.nn.Module] = None,
    teacher: Optional[BPGridTeacher] = None,
    source: Optional[Sequence[str]] = None,
) -> Tuple[List[FrozenPolicyBatch], Dict[str, Any]]:
    """Rebuild ``FrozenPolicyBatch``es from a bank transition (no Episode).

    This is the "independent components of ``frozen_policy_p_solver`` + the
    ``eval_bank`` transition" route (option (a) of the design note): the
    Episode-only helpers ``_build_policy_m_lists`` /
    ``_expand_policy_expectation_children`` are replaced by the exact-eta
    expanded children, clipped/raw M and normalized child weights that
    ``build_transition_for_checkpoint`` already produces -- the same structures
    those Episode helpers build during training.  The frozen residual math
    itself stays the training-side code path (``build_frozen_branch``).
    """
    model = model if model is not None else cp.models["policy_value"]
    q_model = q_model if q_model is not None else model
    teacher = teacher if teacher is not None else build_grid_teacher(
        cp,
        target_model=resolve_teacher_target_model(cp, teacher_target=config.teacher_target),
        q_target_model=q_model,
        max_expanded_states=config.teacher_max_expanded_states,
    )
    p0_loss, pi_loss = build_bellman_losses(cp.economic_config)
    device = _module_device(model)
    parent = transition.parent_states.to(device)
    children_list, m_used, m_raw, weights = split_transition(transition, device=device)
    n_rows = int(parent.shape[0])

    # Both calibers are always reported: residual_tensors exposes the physical
    # residual plus ``R / equity_value_scale`` (the normalized caliber).
    has_scale = callable(getattr(model, "equity_value_scale", None))
    if config.normalize_by_value_scale is not None:
        normalize = bool(config.normalize_by_value_scale) and has_scale
    else:
        normalize = has_scale

    if source is None:
        source_ids = torch.full((n_rows,), -1, dtype=torch.long, device=device)
    else:
        source_ids = torch.tensor(
            [
                0 if label == "simulated" else 1 if label == "coverage" else -1
                for label in source
            ],
            dtype=torch.long,
            device=device,
        )

    problem: List[FrozenPolicyBatch] = []
    parity_errors: List[float] = []
    parity_rows: List[Dict[str, Any]] = []
    for start, stop in iter_row_chunks(n_rows, config.parent_chunk_size):
        chunk = slice(start, stop)
        child_chunk = [item[chunk].to(device) for item in children_list]
        p0_branch, err0, ok0 = build_frozen_branch(
            teacher=teacher,
            model=model,
            q_model=q_model,
            p0_loss=p0_loss,
            pi_loss=pi_loss,
            parent=parent[chunk].to(device),
            children=child_chunk,
            m_used=m_used[chunk].to(device),
            raw_m=m_raw[chunk].to(device),
            child_weights=weights[chunk].to(device),
            branch="p0",
            normalize_by_value_scale=normalize,
        )
        pi_branch, erri, oki = build_frozen_branch(
            teacher=teacher,
            model=model,
            q_model=q_model,
            p0_loss=p0_loss,
            pi_loss=pi_loss,
            parent=parent[chunk].to(device),
            children=child_chunk,
            m_used=m_used[chunk].to(device),
            raw_m=m_raw[chunk].to(device),
            child_weights=weights[chunk].to(device),
            branch="pi",
            normalize_by_value_scale=normalize,
        )
        problem.append(
            FrozenPolicyBatch(
                batch_id=len(problem),
                p0=p0_branch,
                pi=pi_branch,
                parity_max_abs_error=max(err0, erri),
                parity_passed=bool(ok0 and oki),
                source_id=source_ids[chunk],
            )
        )
        parity_errors.extend([err0, erri])
        parity_rows.append(
            {
                "batch_id": len(problem) - 1,
                "row_start": int(start),
                "row_stop": int(stop),
                "parity_max_abs_error_p0": float(err0),
                "parity_max_abs_error_pi": float(erri),
                "parity_passed": float(bool(ok0 and oki)),
                "parity_failing_rows_p0": len(p0_branch.parity_error_rows),
                "parity_failing_rows_pi": len(pi_branch.parity_error_rows),
            }
        )
    diagnostics = {
        "parity_max_abs_error": float(max(parity_errors)) if parity_errors else 0.0,
        "parity_passed": bool(
            all(row["parity_passed"] > 0.5 for row in parity_rows)
        )
        if parity_rows
        else False,
        "parity_tolerance_atol": 1e-5,
        "parity_tolerance_rtol": 1e-5,
        "normalize_by_value_scale": bool(normalize),
        "n_batches": len(problem),
    }
    return problem, {"parity": pd.DataFrame(parity_rows), **diagnostics}


def _collect_frozen_rows(
    model: torch.nn.Module,
    problem: Sequence[FrozenPolicyBatch],
) -> Dict[str, Any]:
    """Per-row frozen-policy and fresh-greedy residuals in both calibers."""
    frozen_stats, frozen_rows = evaluate_problem_rows(model, problem)
    fresh_physical_p0: List[torch.Tensor] = []
    fresh_physical_pi: List[torch.Tensor] = []
    fresh_norm_p0: List[torch.Tensor] = []
    fresh_norm_pi: List[torch.Tensor] = []
    scale_rows: List[torch.Tensor] = []
    p0_rows: List[torch.Tensor] = []
    pi_rows: List[torch.Tensor] = []
    bp0_rows: List[torch.Tensor] = []
    bpi_rows: List[torch.Tensor] = []
    value0_rows: List[torch.Tensor] = []
    valuei_rows: List[torch.Tensor] = []
    for batch in problem:
        p0_online, pi_online = model._value_outputs(batch.parent)
        scale = batch.p0.scale
        fresh0 = p0_online - batch.p0.production_value
        freshi = pi_online - batch.pi.production_value
        fresh_physical_p0.append(fresh0.reshape(-1).detach().cpu())
        fresh_physical_pi.append(freshi.reshape(-1).detach().cpu())
        fresh_norm_p0.append((fresh0 / scale.clamp_min(1e-12)).reshape(-1).detach().cpu())
        fresh_norm_pi.append((freshi / scale.clamp_min(1e-12)).reshape(-1).detach().cpu())
        scale_rows.append(scale.reshape(-1).detach().cpu())
        p0_rows.append(p0_online.reshape(-1).detach().cpu())
        pi_rows.append(pi_online.reshape(-1).detach().cpu())
        bp0_rows.append(batch.p0.bp_star.reshape(-1).detach().cpu())
        bpi_rows.append(batch.pi.bp_star.reshape(-1).detach().cpu())
        value0_rows.append(batch.p0.production_value.reshape(-1).detach().cpu())
        valuei_rows.append(batch.pi.production_value.reshape(-1).detach().cpu())

    def _cat(items: List[torch.Tensor]) -> np.ndarray:
        return torch.cat(items).to(torch.float64).cpu().numpy() if items else np.empty(0)

    return {
        "frozen_r0_physical": frozen_rows["p0"].to(torch.float64).cpu().numpy(),
        "frozen_ri_physical": frozen_rows["pi"].to(torch.float64).cpu().numpy(),
        "frozen_r0_normalized": frozen_rows["p0_norm"].to(torch.float64).cpu().numpy(),
        "frozen_ri_normalized": frozen_rows["pi_norm"].to(torch.float64).cpu().numpy(),
        "frozen_row_finite": frozen_rows["row_finite"].cpu().numpy().astype(bool),
        "fresh_r0_physical": _cat(fresh_physical_p0),
        "fresh_ri_physical": _cat(fresh_physical_pi),
        "fresh_r0_normalized": _cat(fresh_norm_p0),
        "fresh_ri_normalized": _cat(fresh_norm_pi),
        "value_scale": _cat(scale_rows),
        "p0_output_physical": _cat(p0_rows),
        "pi_output_physical": _cat(pi_rows),
        "bp0_star": _cat(bp0_rows),
        "bpi_star": _cat(bpi_rows),
        "value_star_p0": _cat(value0_rows),
        "value_star_pi": _cat(valuei_rows),
        "frozen_stats": frozen_stats,
    }


def _maybe_grid_shape(values: np.ndarray, bank: Mapping[str, Any]) -> np.ndarray:
    """Reshape row vectors to (b, z) when the bank is a synthetic stress grid."""
    if values.ndim != 1:
        return values
    n_rows = int(values.shape[0])
    parent_states = bank.get("parent_states")
    if parent_states is None:
        return values
    states = np.asarray(
        parent_states.detach().cpu() if torch.is_tensor(parent_states) else parent_states,
        dtype=np.float64,
    )
    if states.ndim != 2 or states.shape[0] != n_rows:
        return values
    b_unique = np.unique(states[:, 0])
    z_unique = np.unique(states[:, 1])
    if len(b_unique) * len(z_unique) != n_rows:
        return values
    # Verify the row order is the (b, z) cartesian product used by Bank C.
    expected = np.stack(
        np.meshgrid(b_unique, z_unique, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    if not np.allclose(states[:, :2], expected, atol=1e-6):
        return values
    return values.reshape(len(b_unique), len(z_unique))


def _evaluate_grid_chunked(
    teacher: BPGridTeacher,
    parent_state: torch.Tensor,
    children: Sequence[torch.Tensor],
    m_list: torch.Tensor,
    bp_grid: torch.Tensor,
    *,
    branch: str,
    child_weights: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """``teacher._evaluate_grid`` under the parent chunk hard cap.

    Mirrors ``training.frozen_policy_p_solver._evaluate_grid_chunked`` so the
    evaluator can evaluate extra candidate grids (dense discretization
    diagnostics, objective slices) without violating the expanded-state cap.
    """
    device = _module_device(teacher.target_model)
    parent_state = parent_state.to(device)
    children = [item.to(device) for item in children] if not torch.is_tensor(children) else children.to(device)
    m_list = m_list.to(device)
    bp_grid = bp_grid.to(device)
    if child_weights is not None:
        child_weights = child_weights.to(device)
    n_parent = int(parent_state.shape[0])
    n_children = int(len(children))
    plan = resolve_grid_chunk_plan(
        n_parent=n_parent,
        n_grid=int(bp_grid.shape[1]),
        n_children=n_children,
        configured_parent_chunk=n_parent,
        configured_candidate_chunk=teacher.candidate_chunk_size,
        max_expanded_states=teacher.max_expanded_states,
    )
    chunk = max(1, int(plan.parent_chunk_effective))
    if chunk >= n_parent:
        return teacher._evaluate_grid(
            parent_state, children, m_list, bp_grid,
            branch=branch, child_weights=child_weights,
        )
    outputs = []
    for start in range(0, n_parent, chunk):
        stop = min(start + chunk, n_parent)
        weight_chunk = (
            child_weights
            if child_weights is not None and child_weights.ndim == 1
            else child_weights[start:stop] if child_weights is not None else None
        )
        outputs.append(
            teacher._evaluate_grid(
                parent_state[start:stop],
                _slice_child_axis(children, start, stop),
                _slice_child_axis(m_list, start, stop),
                bp_grid[start:stop],
                branch=branch,
                child_weights=weight_chunk,
            )
        )
    keys = outputs[0].keys()
    return {key: torch.cat([out[key] for out in outputs], dim=0) for key in keys}


def _dense_grid_discretization(
    teacher: BPGridTeacher,
    transition: CheckpointTransition,
    config: BellmanEvalConfig,
) -> pd.DataFrame:
    """Grid-argmax vs dense-candidate-grid argmax gap (evaluation only).

    The dense grid never feeds back into any policy; it is a pure discretization
    diagnostic of the coarse+fine Grid BP search.  eta = 0 rows are forced
    (flat objective) and are flagged rather than mixed into the gap statistics.
    """
    device = _module_device(teacher.target_model)
    parent = transition.parent_states.to(device)
    children_list, m_used, _m_raw, weights = split_transition(transition, device=device)
    n_rows = int(parent.shape[0])
    dense = torch.linspace(
        float(teacher.grid_min),
        float(teacher.grid_max),
        steps=int(config.dense_grid_size),
        dtype=parent.dtype,
        device=parent.device,
    ).reshape(1, -1)

    records: List[Dict[str, Any]] = []
    for start, stop in iter_row_chunks(n_rows, config.parent_chunk_size):
        chunk = slice(start, stop)
        parent_chunk = parent[chunk].to(device)
        dense_grid = dense.expand(parent_chunk.shape[0], dense.shape[1])
        values = _evaluate_grid_chunked(
            teacher,
            parent_chunk,
            [item[chunk].to(device) for item in children_list],
            m_used[chunk].to(device),
            dense_grid,
            branch="p0",
            child_weights=weights[chunk].to(device),
        )
        value_grid = values["value_grid"]
        argmax = value_grid.argmax(dim=1, keepdim=True)
        dense_star = torch.gather(dense_grid, 1, argmax).reshape(-1)
        dense_value = torch.gather(value_grid, 1, argmax).reshape(-1)
        # Compare against the teacher's own coarse+fine star on the same rows.
        solved = teacher.compute_value_target(
            parent_state=parent_chunk,
            children=[item[chunk].to(device) for item in children_list],
            m_list=m_used[chunk].to(device),
            branch="p0",
            child_weights=weights[chunk].to(device),
        )
        grid_star = solved["bp_star"].reshape(-1)
        grid_value = solved["value_star"].reshape(-1)
        for row in range(parent_chunk.shape[0]):
            eta_value = float(parent_chunk[row, 2])
            records.append(
                {
                    "row_index": int(start + row),
                    "eta": eta_value,
                    "refi_active": 1.0 if eta_value > 0.5 else 0.0,
                    "bp_star_grid": float(grid_star[row]),
                    "bp_dense_star": float(dense_star[row]),
                    "bp_gap_abs": float((grid_star[row] - dense_star[row]).abs()),
                    "value_star_grid": float(grid_value[row]),
                    "value_dense_star": float(dense_value[row]),
                    "value_gap": float((dense_value[row] - grid_value[row]).clamp_min(0.0)),
                }
            )
    return pd.DataFrame(records)


def evaluate_firm_bellman(
    cp: EvalCheckpoint,
    bank: Mapping[str, Any],
    *,
    config: Optional[BellmanEvalConfig] = None,
) -> BellmanEvalResult:
    """Evaluate the firm Bellman evidence of ``cp`` on one fixed bank entry.

    ``bank`` is one bank payload of ``evaluation/convergence/eval_bank.py``
    (``parent_states`` / ``hatc_cal`` / ``lnk_cal`` / ``shock_bank`` / ...); the
    children used by every residual are rebuilt from its fixed shocks through
    the checkpoint's own FC1/SDF.
    """
    config = config if config is not None else BellmanEvalConfig()
    model = cp.models["policy_value"]
    device = _module_device(model)
    n_rows = int(torch.as_tensor(bank["parent_states"]).shape[0])
    labels = source_labels(bank, n_rows)
    recovery_mode, recovery_source = _resolve_recovery_mode(cp, config)
    regime_mode = resolve_q_parent_default_regime_mode(config.q_parent_default_regime_mode)

    with torch.inference_mode():
        for module in cp.models.values():
            module.eval()
        transition = build_transition_for_checkpoint(
            _bank_payload_on_device(bank, device),
            cp.models["sdf_fc1"],
            cp.hyperparams,
            cp.economic_config,
        )
        parent = transition.parent_states.to(device)
        teacher = build_grid_teacher(
            cp,
            target_model=resolve_teacher_target_model(cp, teacher_target=config.teacher_target),
            max_expanded_states=config.teacher_max_expanded_states,
        )
        problem, problem_diagnostics = build_frozen_policy_problem(
            cp,
            transition,
            config=config,
            model=model,
            teacher=teacher,
            source=labels,
        )
        collected = _collect_frozen_rows(model, problem)

        # ---- fresh-greedy residual (explicit re-solve variant) ----
        if config.fresh_greedy_source == "resolve":
            children_list, m_used, _m_raw, weights = split_transition(transition, device=device)
            fresh_p0: List[torch.Tensor] = []
            fresh_pi: List[torch.Tensor] = []
            fresh_physical_p0: List[torch.Tensor] = []
            fresh_physical_pi: List[torch.Tensor] = []
            for start, stop in iter_row_chunks(n_rows, config.parent_chunk_size):
                chunk = slice(start, stop)
                scale = (
                    model.equity_value_scale(parent[chunk]).clamp_min(1e-12)
                    if callable(getattr(model, "equity_value_scale", None))
                    else torch.ones((stop - start, 1), device=parent.device, dtype=parent.dtype)
                )
                p0_online, pi_online = model._value_outputs(parent[chunk])
                for branch, online, sink, phys_sink in (
                    ("p0", p0_online, fresh_p0, fresh_physical_p0),
                    ("pi", pi_online, fresh_pi, fresh_physical_pi),
                ):
                    target = teacher.compute_value_target(
                        parent_state=parent[chunk].to(device),
                        children=[item[chunk].to(device) for item in children_list],
                        m_list=m_used[chunk].to(device),
                        branch=branch,
                        child_weights=weights[chunk].to(device),
                    )
                    physical = online - target["value_star"]
                    phys_sink.append(physical.reshape(-1).detach().cpu())
                    sink.append((physical / scale).reshape(-1).detach().cpu())
            collected["fresh_r0_normalized"] = torch.cat(fresh_p0).to(torch.float64).cpu().numpy()
            collected["fresh_ri_normalized"] = torch.cat(fresh_pi).to(torch.float64).cpu().numpy()
            collected["fresh_r0_physical"] = torch.cat(fresh_physical_p0).to(torch.float64).cpu().numpy()
            collected["fresh_ri_physical"] = torch.cat(fresh_physical_pi).to(torch.float64).cpu().numpy()

        # ---- Q calibers and targets ----
        calibers = q_caliber_tensors(
            model,
            parent,
            economic_config=cp.economic_config,
            recovery_normalization_mode=recovery_mode,
            chunk_size=config.chunk_size,
        )
        targets = {
            m_mode: q_bellman_targets(
                model,
                transition,
                economic_config=cp.economic_config,
                q_calibers=calibers,
                m_mode=m_mode,
                recovery_normalization_mode=recovery_mode,
                parent_default_regime_mode=regime_mode,
                parent_default_eps=config.q_parent_default_eps,
                parent_default_tau=config.q_parent_default_tau,
            )
            for m_mode in ("trainM", "rawM")
        }

        # ---- dense BP candidate grid discretization error ----
        discretization = _dense_grid_discretization(teacher, transition, config)

    parent_np = parent.detach().to(torch.float64).cpu().numpy()
    eta = parent_np[:, 2]
    b = parent_np[:, 0]
    z = parent_np[:, 1]
    scale_np = collected["value_scale"]
    phat_np = calibers["Phat"].detach().to(torch.float64).cpu().numpy().reshape(-1)
    bar_z_np = calibers["bar_z"].detach().to(torch.float64).cpu().numpy().reshape(-1)

    regions = q_region_masks(
        b,
        phat_np,
        bar_z_np,
        zero_b_eps=config.zero_b_eps,
        phat_eps=config.q_boundary_phat_eps,
        bar_z_low=config.q_boundary_bar_z_low,
        bar_z_high=config.q_boundary_bar_z_high,
    )

    metrics: Dict[str, float] = {}
    p0_level = np.abs(collected["p0_output_physical"])
    pi_level = np.abs(collected["pi_output_physical"])

    def _block(values: np.ndarray, prefix: str, level: np.ndarray) -> None:
        metrics.update(
            residual_stat_block(
                values, prefix=prefix, level=level, eta=eta, source=labels
            )
        )

    _block(collected["frozen_r0_physical"], "frozen_p0_physical", p0_level)
    _block(
        collected["frozen_r0_normalized"],
        "frozen_p0_normalized",
        p0_level / np.maximum(scale_np, 1e-12),
    )
    _block(collected["frozen_ri_physical"], "frozen_pi_physical", pi_level)
    _block(
        collected["frozen_ri_normalized"],
        "frozen_pi_normalized",
        pi_level / np.maximum(scale_np, 1e-12),
    )
    _block(collected["fresh_r0_physical"], "fresh_p0_physical", p0_level)
    _block(
        collected["fresh_r0_normalized"],
        "fresh_p0_normalized",
        p0_level / np.maximum(scale_np, 1e-12),
    )
    _block(collected["fresh_ri_physical"], "fresh_pi_physical", pi_level)
    _block(
        collected["fresh_ri_normalized"],
        "fresh_pi_normalized",
        pi_level / np.maximum(scale_np, 1e-12),
    )

    # ---- Q region residual blocks (claim / unit / effective / recovery) ----
    q_claim = calibers["Q_claim"].detach().to(torch.float64).cpu().numpy().reshape(-1)
    q_unit = calibers["q_unit"].detach().to(torch.float64).cpu().numpy().reshape(-1)
    q_effective = calibers["Q_effective"].detach().to(torch.float64).cpu().numpy().reshape(-1)
    recovery = calibers["recovery"].detach().to(torch.float64).cpu().numpy().reshape(-1)
    recovery_target = calibers["recovery_target"].detach().to(torch.float64).cpu().numpy().reshape(-1)
    q_region_rows: List[Dict[str, Any]] = []
    for m_mode in ("trainM", "rawM"):
        target = targets[m_mode]
        q_target_bellman = target["q_target_bellman"].detach().to(torch.float64).cpu().numpy().reshape(-1)
        q_target_training = target["q_target_training"].detach().to(torch.float64).cpu().numpy().reshape(-1)
        q_target_settlement = np.where(phat_np <= 0.0, recovery, q_target_bellman)
        unit_target = np.where(b > 0.0, q_target_bellman / np.maximum(b, 1e-12), np.nan)
        error_series = {
            "claim_bellman": (q_claim - q_target_bellman, q_target_bellman),
            "claim_training": (q_claim - q_target_training, q_target_training),
            "unit_bellman": (q_unit - unit_target, unit_target),
            "effective_settlement": (q_effective - q_target_settlement, q_target_settlement),
            "effective_vs_recovery": (q_effective - recovery, recovery),
        }
        for region_name, region_mask in regions.items():
            for caliber, (errors, levels) in error_series.items():
                prefix = f"q_{region_name}_{caliber}_{m_mode}"
                block = residual_stat_block(
                    errors[region_mask],
                    prefix=prefix,
                    level=levels[region_mask],
                    eta=eta[region_mask],
                )
                metrics.update(block)
                q_region_rows.append(
                    {
                        "region": region_name,
                        "caliber": caliber,
                        "m_mode": m_mode,
                        "n": block[f"{prefix}_n"],
                        "n_finite": block[f"{prefix}_n_finite"],
                        "nonfinite_ratio": block[f"{prefix}_nonfinite_ratio"],
                        "mae": block[f"{prefix}_mae"],
                        "rmse": block[f"{prefix}_rmse"],
                        "p90": block[f"{prefix}_p90"],
                        "p99": block[f"{prefix}_p99"],
                        "max": block[f"{prefix}_max"],
                        "relative_mae": block.get(f"{prefix}_relative_mae", float("nan")),
                    }
                )

    for region_name, region_mask in regions.items():
        metrics[f"q_{region_name}_share"] = (
            float(region_mask.mean()) if region_mask.size else float("nan")
        )
        metrics[f"q_{region_name}_n"] = float(region_mask.sum())

    # ---- parity / finiteness / evaluation-only diagnostics ----
    metrics["parity_max_abs_error"] = float(problem_diagnostics["parity_max_abs_error"])
    metrics["parity_passed"] = 1.0 if problem_diagnostics["parity_passed"] else 0.0
    metrics["frozen_all_finite"] = 1.0 if collected["frozen_stats"].get("all_finite") else 0.0
    metrics["frozen_nonfinite_ratio"] = float(
        collected["frozen_stats"].get("nonfinite_ratio", float("nan"))
    )
    metrics["optimizer_steps"] = 0.0

    # ---- dense-grid discretization metrics (refi-active rows only) ----
    discretization_keys = (
        "bp_grid_discretization_bp_gap_mean",
        "bp_grid_discretization_bp_gap_p90",
        "bp_grid_discretization_bp_gap_max",
        "bp_grid_discretization_value_gap_mean",
        "bp_grid_discretization_value_gap_p90",
        "bp_grid_discretization_value_gap_max",
        "bp_grid_discretization_n_eta1",
        "bp_grid_discretization_n_eta0",
    )
    if discretization is not None and not discretization.empty:
        active = discretization["refi_active"] > 0.5
        active_frame = discretization.loc[active]
        bp_gap = active_frame["bp_gap_abs"].to_numpy(dtype=np.float64)
        value_gap = active_frame["value_gap"].to_numpy(dtype=np.float64)
        metrics["bp_grid_discretization_bp_gap_mean"] = (
            float(np.nanmean(bp_gap)) if bp_gap.size else float("nan")
        )
        metrics["bp_grid_discretization_bp_gap_p90"] = (
            float(np.nanquantile(bp_gap, 0.9)) if np.isfinite(bp_gap).any() else float("nan")
        )
        metrics["bp_grid_discretization_bp_gap_max"] = (
            float(np.nanmax(bp_gap)) if np.isfinite(bp_gap).any() else float("nan")
        )
        metrics["bp_grid_discretization_value_gap_mean"] = (
            float(np.nanmean(value_gap)) if value_gap.size else float("nan")
        )
        metrics["bp_grid_discretization_value_gap_p90"] = (
            float(np.nanquantile(value_gap, 0.9)) if np.isfinite(value_gap).any() else float("nan")
        )
        metrics["bp_grid_discretization_value_gap_max"] = (
            float(np.nanmax(value_gap)) if np.isfinite(value_gap).any() else float("nan")
        )
        metrics["bp_grid_discretization_n_eta1"] = float(active_frame.shape[0])
        metrics["bp_grid_discretization_n_eta0"] = float((~active).sum())
    else:
        for key in discretization_keys:
            metrics[key] = float("nan")

    # ---- row table ----
    rows_table = pd.DataFrame(
        {
            "row_index": np.arange(n_rows, dtype=np.int64),
            "b": b,
            "z": z,
            "eta": eta,
            "i": parent_np[:, 3],
            "x": parent_np[:, 4],
            "source_label": labels,
            "value_scale": scale_np,
            "p0_physical": collected["p0_output_physical"],
            "pi_physical": collected["pi_output_physical"],
            "p0_normalized": collected["p0_output_physical"] / np.maximum(scale_np, 1e-12),
            "pi_normalized": collected["pi_output_physical"] / np.maximum(scale_np, 1e-12),
            "frozen_r0_physical": collected["frozen_r0_physical"],
            "frozen_r0_normalized": collected["frozen_r0_normalized"],
            "frozen_ri_physical": collected["frozen_ri_physical"],
            "frozen_ri_normalized": collected["frozen_ri_normalized"],
            "fresh_r0_physical": collected["fresh_r0_physical"],
            "fresh_r0_normalized": collected["fresh_r0_normalized"],
            "fresh_ri_physical": collected["fresh_ri_physical"],
            "fresh_ri_normalized": collected["fresh_ri_normalized"],
            "frozen_row_finite": collected["frozen_row_finite"],
            "bp0_star": collected["bp0_star"],
            "bpi_star": collected["bpi_star"],
            "value_star_p0": collected["value_star_p0"],
            "value_star_pi": collected["value_star_pi"],
            "Phat": phat_np,
            "P": calibers["P"].detach().to(torch.float64).cpu().numpy().reshape(-1),
            "bar_z": bar_z_np,
            "q_unit": q_unit,
            "Q_claim": q_claim,
            "Q_effective": q_effective,
            "recovery": recovery,
            "recovery_target": recovery_target,
            "q_target_bellman_trainM": targets["trainM"]["q_target_bellman"].detach().to(torch.float64).cpu().numpy().reshape(-1),
            "q_target_bellman_rawM": targets["rawM"]["q_target_bellman"].detach().to(torch.float64).cpu().numpy().reshape(-1),
            "rq_bellman_trainM": targets["trainM"]["rq_bellman_signed"].detach().to(torch.float64).cpu().numpy().reshape(-1),
            "rq_bellman_rawM": targets["rawM"]["rq_bellman_signed"].detach().to(torch.float64).cpu().numpy().reshape(-1),
            "q_target_training_trainM": targets["trainM"]["q_target_training"].detach().to(torch.float64).cpu().numpy().reshape(-1),
            "rq_training_trainM": targets["trainM"]["rq_training_signed"].detach().to(torch.float64).cpu().numpy().reshape(-1),
            "region_q0": regions["q0"],
            "region_qd": regions["qd"],
            "region_qs": regions["qs"],
            "region_qboundary": regions["qboundary"],
        }
    )

    residual_stats_rows = [
        {"metric": key, "value": value}
        for key, value in sorted(metrics.items())
        if key.startswith(("frozen_", "fresh_"))
    ]

    surfaces = {
        "P0_physical": collected["p0_output_physical"],
        "PI_physical": collected["pi_output_physical"],
        "P0_normalized": collected["p0_output_physical"] / np.maximum(scale_np, 1e-12),
        "PI_normalized": collected["pi_output_physical"] / np.maximum(scale_np, 1e-12),
        "P": calibers["P"].detach().to(torch.float64).cpu().numpy().reshape(-1),
        "Phat": phat_np,
        "bar_z": bar_z_np,
        "q_unit": q_unit,
        "Q_claim": q_claim,
        "Q_effective": q_effective,
        "recovery": recovery,
        "R0_frozen_physical": collected["frozen_r0_physical"],
        "R0_frozen_normalized": collected["frozen_r0_normalized"],
        "RI_frozen_physical": collected["frozen_ri_physical"],
        "RI_frozen_normalized": collected["frozen_ri_normalized"],
        "R0_fresh_physical": collected["fresh_r0_physical"],
        "R0_fresh_normalized": collected["fresh_r0_normalized"],
        "RI_fresh_physical": collected["fresh_ri_physical"],
        "RI_fresh_normalized": collected["fresh_ri_normalized"],
        "Q_target_bellman_trainM": targets["trainM"]["q_target_bellman"].detach().to(torch.float64).cpu().numpy().reshape(-1),
        "Q_target_bellman_rawM": targets["rawM"]["q_target_bellman"].detach().to(torch.float64).cpu().numpy().reshape(-1),
        "RQ_bellman_trainM": targets["trainM"]["rq_bellman_signed"].detach().to(torch.float64).cpu().numpy().reshape(-1),
        "RQ_bellman_rawM": targets["rawM"]["rq_bellman_signed"].detach().to(torch.float64).cpu().numpy().reshape(-1),
        "bp0_star": collected["bp0_star"],
        "bpi_star": collected["bpi_star"],
        "value_star_p0": collected["value_star_p0"],
        "value_star_pi": collected["value_star_pi"],
    }
    surfaces = {name: _maybe_grid_shape(values, bank) for name, values in surfaces.items()}

    record = semantic_record(cp)
    diagnostics: Dict[str, Any] = {
        "caliber_note": CALIBER_NOTE,
        "value_parameterization": dict(record.get("value_parameterization") or {}),
        "q_parameterization": record.get("q_parameterization"),
        "q_recovery_normalization_mode": recovery_mode,
        "q_recovery_normalization_mode_source": recovery_source,
        "q_parent_default_regime_mode": regime_mode,
        "m_calibers": {
            "trainM": transition.metadata.get("m_mode"),
            "rawM": "raw_sdf_m",
            "declared": "both trainM and rawM are reported; canonical alias would be trainM",
        },
        "residual_definitions": {
            "frozen": (
                "R0 = P0_online - CF0_frozen - sum_j w_j M_j P_child_online_j; "
                "RI = PI_online - CFI_frozen - G * sum_j w_j M_j P_child_online_j "
                "(policy frozen at build time: greedy bp_star / selector / survival / Q)"
            ),
            "fresh": (
                "(P_online - value_star) where value_star is the re-solved greedy "
                f"Grid-BP RHS of the current model (fresh_greedy_source={config.fresh_greedy_source})"
            ),
        },
        "teacher_target": config.teacher_target,
        "teacher_grid": {
            "grid_min": teacher.grid_min,
            "grid_max": teacher.grid_max,
            "coarse_size": teacher.coarse_size,
            "fine_size": teacher.fine_size,
            "refine": teacher.refine,
            "quadratic_refine": teacher.quadratic_refine,
            "max_expanded_states": teacher.max_expanded_states,
        },
        "parity": {
            "max_abs_error": float(problem_diagnostics["parity_max_abs_error"]),
            "passed": bool(problem_diagnostics["parity_passed"]),
            "atol": 1e-5,
            "rtol": 1e-5,
            "note": "parity (rhs_snapshot vs teacher value_star) is a diagnostic, not an acceptance gate",
        },
        "transition_metadata": dict(transition.metadata),
        "bank_id": bank.get("bank_id"),
        "bank_tags": list(bank.get("tags") or []),
        "episode": cp.episode,
        "parent_states_sha256": content_sha256(parent.detach().cpu()),
        "evaluation_only": True,
        "optimizer_steps": 0,
        "model_mode": "eval",
        "frozen_training_side_stats": {
            key: value
            for key, value in collected["frozen_stats"].items()
            if not isinstance(value, torch.Tensor)
        },
    }

    tables = {
        "rows": rows_table,
        "residual_stats": pd.DataFrame(residual_stats_rows),
        "q_region_stats": pd.DataFrame(q_region_rows),
        "bp_discretization": discretization,
        "parity": problem_diagnostics["parity"],
    }
    return BellmanEvalResult(
        metrics=metrics,
        tables=tables,
        surfaces=surfaces,
        diagnostics=diagnostics,
    )
