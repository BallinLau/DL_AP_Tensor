"""Read-only P-teacher drift decomposition across episode checkpoints.

This evaluator intentionally does one job: explain changes in the current
checkpoint Bellman P teacher through production, investment, refinancing,
equity-financing cost, continuation, and P0/PI branch selection.  It does not
run training, simulation, FC1 validation, or BP-consistency orchestration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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
from evaluation.convergence_artifacts import (  # noqa: E402
    discover_episode_firm_data,
    parse_episode_selection,
)
from evaluation.full_run_diagnostics import model_state_hash  # noqa: E402
from evaluation.grids import ReferenceFirmState, load_reference_state  # noqa: E402
from losses import P0Loss, PILoss  # noqa: E402
from losses.utils import compute_cashflow  # noqa: E402
from training.bp_grid_teacher import BPGridTeacher  # noqa: E402


DEFAULT_B_VALUES = (0.01, 0.05, 0.20)
DEFAULT_Z_VALUES = (0.0, 2.0, 3.0, 4.0)
DEFAULT_ETA_VALUES = (0.0, 1.0)
IDENTITY_TOL = 1e-5
ETA_ZERO_TOL = 1e-7
CHECKPOINT_RE = re.compile(r"^ep(?P<episode>\d+)_combined\.pt$")
ERROR_COLUMNS = ("episode", "block", "error", "traceback")


def select_branch_values(
    t0: np.ndarray, ti: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select PI only for strict TI > T0, matching max(T0, TI) tie semantics."""
    t0 = np.asarray(t0, dtype=np.float64)
    ti = np.asarray(ti, dtype=np.float64)
    selected_is_pi = ti > t0
    return np.where(selected_is_pi, ti, t0), selected_is_pi


def reconstruct_p0(
    production: np.ndarray,
    financing: np.ndarray,
    equity_cost: np.ndarray,
    continuation: np.ndarray,
) -> np.ndarray:
    return production + financing - equity_cost + continuation


def reconstruct_pi(
    production: np.ndarray,
    investment: np.ndarray,
    financing: np.ndarray,
    equity_cost: np.ndarray,
    continuation: np.ndarray,
) -> np.ndarray:
    return production - investment + financing - equity_cost + continuation


def aggregate_selected_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the strict selected branch over the production i-grid."""
    rows: List[Dict[str, Any]] = []
    for keys, group in frame.groupby(["episode", "eta", "b", "z"], sort=True):
        episode, eta, b_value, z_value = keys
        finite = group["selected_finite"].astype(bool).to_numpy()
        selected_finite_share = float(finite.mean()) if len(finite) else float("nan")

        def mean(column: str) -> float:
            values = group[column].to_numpy(dtype=np.float64)
            return float(np.mean(values)) if np.isfinite(values).all() else float("nan")

        phat_teacher = mean("T_selected")
        production_component = mean("production")
        investment_component = -mean("investment_selected")
        financing_component = mean("financing_selected")
        equity_cost_component = -mean("equity_cost_selected")
        continuation_component = mean("continuation_selected")
        reconstructed = (
            production_component
            + investment_component
            + financing_component
            + equity_cost_component
            + continuation_component
        )
        bp_values = group["bp_selected"].to_numpy(dtype=np.float64)
        q_issue_values = group["q_issue_selected"].to_numpy(dtype=np.float64)
        rows.append({
            "episode": int(episode),
            "eta": float(eta),
            "b": float(b_value),
            "z": float(z_value),
            "Phat_pred": float(group["Phat_pred"].iloc[0]),
            "P_pred": float(group["P_pred"].iloc[0]),
            "bar_z_pred": float(group["bar_z_pred"].iloc[0]),
            "Phat_teacher": phat_teacher,
            "production_component": production_component,
            "investment_component": investment_component,
            "financing_component": financing_component,
            "equity_cost_component": equity_cost_component,
            "continuation_component": continuation_component,
            "pi_selected_share": float(group["selected_is_pi"].mean()),
            "bp_selected_mean": (
                float(np.mean(bp_values)) if np.isfinite(bp_values).all() else float("nan")
            ),
            "bp_selected_p50": (
                float(np.median(bp_values)) if np.isfinite(bp_values).all() else float("nan")
            ),
            "q_current_claim": float(group["q_current_claim"].iloc[0]),
            "q_issue_selected_mean": (
                float(np.mean(q_issue_values))
                if np.isfinite(q_issue_values).all()
                else float("nan")
            ),
            "phat_pred_minus_teacher": float(group["Phat_pred"].iloc[0]) - phat_teacher,
            "phat_identity_error": phat_teacher - reconstructed,
            "selected_finite_share": selected_finite_share,
        })
    return pd.DataFrame(rows)


DRIFT_COMPONENTS = {
    "Phat_teacher": "delta_phat_teacher",
    "production_component": "delta_production",
    "investment_component": "delta_investment",
    "financing_component": "delta_financing",
    "equity_cost_component": "delta_equity_cost",
    "continuation_component": "delta_continuation",
    "pi_selected_share": "delta_pi_selected_share",
}
DRIFT_COLUMNS = (
    "episode_from", "episode_to", "eta", "b", "z",
    *DRIFT_COMPONENTS.values(), "drift_reconstructed", "drift_identity_error",
)
CUMULATIVE_COLUMNS = (
    "episode_from", "episode_to", "eta", "b", "z",
    "cumulative_phat_drift", "cumulative_production_contribution",
    "cumulative_investment_contribution", "cumulative_financing_contribution",
    "cumulative_equity_cost_contribution", "cumulative_continuation_contribution",
    "cumulative_pi_selected_share_change", "cumulative_reconstructed",
    "cumulative_identity_error", "financing_ratio", "continuation_ratio",
    "investment_ratio", "equity_cost_ratio", "production_ratio", "mechanism_label",
)


def compute_episode_drift(levels: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, group in levels.groupby(["eta", "b", "z"], sort=True):
        group = group.sort_values("episode")
        records = list(group.to_dict("records"))
        for previous, current in zip(records[:-1], records[1:]):
            row: Dict[str, Any] = {
                "episode_from": int(previous["episode"]),
                "episode_to": int(current["episode"]),
                "eta": float(keys[0]),
                "b": float(keys[1]),
                "z": float(keys[2]),
            }
            for source, target in DRIFT_COMPONENTS.items():
                left = float(previous[source])
                right = float(current[source])
                row[target] = right - left if np.isfinite([left, right]).all() else float("nan")
            row["drift_reconstructed"] = sum(
                row[name]
                for name in (
                    "delta_production",
                    "delta_investment",
                    "delta_financing",
                    "delta_equity_cost",
                    "delta_continuation",
                )
            )
            row["drift_identity_error"] = (
                row["delta_phat_teacher"] - row["drift_reconstructed"]
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=DRIFT_COLUMNS)


def compute_cumulative_drift(
    levels: pd.DataFrame, *, episode_from: int = 1, episode_to: int = 4
) -> pd.DataFrame:
    before = levels.loc[levels["episode"] == episode_from]
    after = levels.loc[levels["episode"] == episode_to]
    if before.empty or after.empty:
        return pd.DataFrame(columns=CUMULATIVE_COLUMNS)
    merged = before.merge(after, on=["eta", "b", "z"], suffixes=("_from", "_to"))
    rows: List[Dict[str, Any]] = []
    for record in merged.to_dict("records"):
        row: Dict[str, Any] = {
            "episode_from": episode_from,
            "episode_to": episode_to,
            "eta": float(record["eta"]),
            "b": float(record["b"]),
            "z": float(record["z"]),
        }
        for source, target in (
            ("Phat_teacher", "cumulative_phat_drift"),
            ("production_component", "cumulative_production_contribution"),
            ("investment_component", "cumulative_investment_contribution"),
            ("financing_component", "cumulative_financing_contribution"),
            ("equity_cost_component", "cumulative_equity_cost_contribution"),
            ("continuation_component", "cumulative_continuation_contribution"),
            ("pi_selected_share", "cumulative_pi_selected_share_change"),
        ):
            left, right = float(record[f"{source}_from"]), float(record[f"{source}_to"])
            row[target] = right - left if np.isfinite([left, right]).all() else float("nan")
        row["cumulative_reconstructed"] = sum(
            row[name]
            for name in (
                "cumulative_production_contribution",
                "cumulative_investment_contribution",
                "cumulative_financing_contribution",
                "cumulative_equity_cost_contribution",
                "cumulative_continuation_contribution",
            )
        )
        row["cumulative_identity_error"] = (
            row["cumulative_phat_drift"] - row["cumulative_reconstructed"]
        )
        denominator = row["cumulative_phat_drift"]
        contribution_names = {
            "financing_ratio": "cumulative_financing_contribution",
            "continuation_ratio": "cumulative_continuation_contribution",
            "investment_ratio": "cumulative_investment_contribution",
            "equity_cost_ratio": "cumulative_equity_cost_contribution",
            "production_ratio": "cumulative_production_contribution",
        }
        valid = abs(denominator) > 1e-8 and all(
            np.isfinite(row[column]) for column in contribution_names.values()
        )
        for ratio, contribution in contribution_names.items():
            row[ratio] = row[contribution] / denominator if valid else float("nan")
        row["mechanism_label"] = classify_mechanism(
            row["cumulative_financing_contribution"],
            row["cumulative_continuation_contribution"],
        ) if valid else "UNRESOLVED_NONFINITE"
        rows.append(row)
    return pd.DataFrame(rows, columns=CUMULATIVE_COLUMNS)


def classify_mechanism(financing: float, continuation: float) -> str:
    if not np.isfinite([financing, continuation]).all():
        return "UNRESOLVED_NONFINITE"
    if abs(continuation) > 1.5 * abs(financing):
        return "CONTINUATION_DOMINANT"
    if abs(financing) > 1.5 * abs(continuation):
        return "FINANCING_DOMINANT"
    return "MIXED"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def discover_checkpoints(run_root: Path) -> Dict[int, Path]:
    found: Dict[int, Path] = {}
    for directory in (run_root / "checkpoints_analysis", run_root / "checkpoints"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("ep*_combined.pt")):
            match = CHECKPOINT_RE.match(path.name)
            if match:
                found.setdefault(int(match.group("episode")), path.resolve())
    return dict(sorted(found.items()))


def _macro_for_firm(firm: Path) -> Path | None:
    candidates = [firm.with_name(f"{firm.stem}_macro{firm.suffix}")]
    if firm.stem.endswith("_firm"):
        candidates.insert(
            0,
            firm.with_name(f"{firm.stem.removesuffix('_firm')}_macro{firm.suffix}"),
        )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def choose_reference_artifacts(run_root: Path) -> tuple[Path, Path | None]:
    final_firm = run_root / "data" / "outputs" / "final_simulate_firm.pkl"
    if final_firm.is_file():
        return final_firm.resolve(), _macro_for_firm(final_firm)
    firms, _warnings = discover_episode_firm_data(run_root)
    if not firms:
        raise FileNotFoundError(f"No episode firm data found under {run_root}")
    firm = firms[max(firms)]
    return firm, _macro_for_firm(firm)


def build_focus_states(
    reference: ReferenceFirmState,
    b_values: Sequence[float],
    z_values: Sequence[float],
    *,
    eta: float,
    i_value: float,
    device: torch.device,
) -> torch.Tensor:
    rows = [
        [b, z, eta, i_value, reference.x, reference.hatcf, reference.lnkf]
        for b in b_values
        for z in z_values
    ]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _losses(loaded) -> tuple[P0Loss, PILoss]:
    config = loaded.economic_config
    p0 = P0Loss(
        delta=config.DELTA,
        tau=config.TAU,
        kappa_b=config.KAPPA_B,
        kappa_e=config.KAPPA_E,
        aio_weight=config.AIO_WEIGHT,
        alpha_z=config.ALPHA_Z,
        beta_z=config.BETA_Z,
        z0=config.Z0,
    )
    pi = PILoss(
        delta=config.DELTA,
        tau=config.TAU,
        g=config.G,
        kappa_b=config.KAPPA_B,
        kappa_e=config.KAPPA_E,
        aio_weight=config.AIO_WEIGHT,
        alpha_z=config.ALPHA_Z,
        beta_z=config.BETA_Z,
        z0=config.Z0,
        b_penalty_weight=0.0,
    )
    return p0, pi


def _tensor_column(value: torch.Tensor) -> np.ndarray:
    return value.detach().reshape(-1).cpu().numpy().astype(np.float64)


def _teacher_branch_components(
    result: Mapping[str, torch.Tensor],
    states: torch.Tensor,
    *,
    branch: str,
    p0_loss: P0Loss,
    pi_loss: PILoss,
    model: torch.nn.Module,
) -> Dict[str, np.ndarray]:
    value = _tensor_column(result["value_star"])
    bp_star = _tensor_column(result["bp_star"])
    q_issue = _tensor_column(result["q_issue_at_star"])
    if "q_current_claim" in result:
        q_current_t = result["q_current_claim"].reshape(-1, 1)
    else:
        output = model(states)
        q_current_t = output["Q"] if isinstance(output, dict) else output.Q
    q_current = _tensor_column(q_current_t)
    production_t = compute_cashflow(
        states[:, 4:5],
        states[:, 1:2],
        states[:, 0:1],
        p0_loss.delta,
        p0_loss.tau,
    )
    if branch == "p0":
        cashflow_t = p0_loss.compute_cashflow_p0(
            states[:, 4:5], states[:, 1:2], states[:, 0:1],
            q_current_t, result["q_issue_at_star"].reshape(-1, 1), states[:, 2:3],
        )
    elif branch == "pi":
        cashflow_t = pi_loss.compute_cashflow_pi(
            states[:, 4:5], states[:, 1:2], states[:, 0:1], states[:, 3:4],
            q_current_t, result["q_issue_at_star"].reshape(-1, 1), states[:, 2:3],
        )
    else:
        raise ValueError(f"unsupported branch: {branch}")
    cashflow = _tensor_column(cashflow_t)
    continuation = value - cashflow
    continuation_source = np.full(value.shape, "value_minus_cashflow", dtype=object)
    continuation_grid = result.get("continuation_grid_mean")
    argmax_index = result.get("argmax_index")
    bp_star_grid = result.get("bp_star_grid")
    if continuation_grid is not None and argmax_index is not None:
        row_index = torch.arange(argmax_index.shape[0], device=argmax_index.device)
        direct = _tensor_column(
            continuation_grid[row_index, argmax_index.reshape(-1)]
        )
        exact_grid_star = np.ones_like(direct, dtype=bool)
        if bp_star_grid is not None:
            exact_grid_star = np.isclose(
                _tensor_column(result["bp_star"]),
                _tensor_column(bp_star_grid),
                rtol=0.0,
                atol=1e-8,
            )
        use_direct = np.isfinite(direct) & exact_grid_star
        continuation[use_direct] = direct[use_direct]
        continuation_source[use_direct] = "teacher_continuation_grid"
    return {
        "value": value,
        "bp_star": bp_star,
        "q_current": q_current,
        "q_issue": q_issue,
        "production": _tensor_column(production_t),
        "cashflow": cashflow,
        "continuation": continuation,
        "continuation_source": continuation_source,
        "p_child": _tensor_column(result["p_child_at_star"]),
        "default_at_star": _tensor_column(result["default_at_star"]),
    }


def _finite_share(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.isfinite(array).mean()) if array.size else float("nan")


def evaluate_episode(
    *,
    episode: int,
    checkpoint: Path,
    reference: ReferenceFirmState,
    b_values: Sequence[float],
    z_values: Sequence[float],
    eta_values: Sequence[float],
    n_child_shocks: int,
    shock_seed: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    started = time.perf_counter()
    print(f"[ep{episode}] load checkpoint ...", flush=True)
    load_start = time.perf_counter()
    loaded = load_analysis_checkpoint(checkpoint, device=device)
    model = loaded.models["policy_value"]
    sdf_fc1 = loaded.models["sdf_fc1"]
    model.eval()
    sdf_fc1.eval()
    policy_hash_before = model_state_hash(model)
    sdf_hash_before = model_state_hash(sdf_fc1)
    load_seconds = time.perf_counter() - load_start

    i_grid_size = max(int(model.i_grid_size), 2)
    i_values = np.linspace(0.0, float(model.i_threshold), i_grid_size, dtype=np.float64)
    warnings: List[str] = []
    errors: List[Dict[str, Any]] = []
    if i_grid_size != 11:
        warnings.append(f"checkpoint i_grid_size={i_grid_size}, expected production value 11")

    base_states = build_focus_states(
        reference, b_values, z_values, eta=0.0,
        i_value=float(reference.i_mid), device=device,
    )
    transition_start = time.perf_counter()
    print(f"[ep{episode}] transition built ...", flush=True)
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
    transition_seconds = time.perf_counter() - transition_start

    p0_loss, pi_loss = _losses(loaded)
    teacher = BPGridTeacher.from_hyperparams(
        model, p0_loss, pi_loss, loaded.hyperparams,
    )
    n_rows = len(b_values) * len(z_values)
    prediction: Dict[float, Dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for eta in eta_values:
            state = build_focus_states(
                reference, b_values, z_values, eta=eta,
                i_value=float(reference.i_mid), device=device,
            )
            phat, p_value, bar_z, _survival = model.cal_phats(state)
            prediction[float(eta)] = {
                "Phat": _tensor_column(phat),
                "P": _tensor_column(p_value),
                "bar_z": _tensor_column(bar_z),
            }

    teacher_start = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    expanded_count = int(transition.stacked_children().shape[1])
    with _checkpoint_economic_config(loaded.economic_config), torch.no_grad():
        for i_index, i_value in enumerate(i_values):
            for eta in eta_values:
                states = build_focus_states(
                    reference, b_values, z_values, eta=eta,
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
                p0_result = bundles[0][expanded_count]
                pi_result = bundles[1][expanded_count]
                p0 = _teacher_branch_components(
                    p0_result, states, branch="p0", p0_loss=p0_loss,
                    pi_loss=pi_loss, model=model,
                )
                pi = _teacher_branch_components(
                    pi_result, states, branch="pi", p0_loss=p0_loss,
                    pi_loss=pi_loss, model=model,
                )
                t_selected, selected_is_pi = select_branch_values(p0["value"], pi["value"])
                financing0 = eta * ((1.0 - p0_loss.kappa_b) * p0["q_issue"] - p0["q_current"])
                financingi = eta * (
                    (1.0 - pi_loss.kappa_b) * pi_loss.g * pi["q_issue"] - pi["q_current"]
                )
                equity0 = p0["production"] + financing0 - p0["cashflow"]
                equityi = p0["production"] - float(i_value) + financingi - pi["cashflow"]
                t0_reconstructed = reconstruct_p0(
                    p0["production"], financing0, equity0, p0["continuation"]
                )
                ti_reconstructed = reconstruct_pi(
                    p0["production"], np.full(n_rows, float(i_value)),
                    financingi, equityi, pi["continuation"],
                )
                for row_index, (b_value, z_value) in enumerate(
                    (pair for b in b_values for pair in ((b, z) for z in z_values))
                ):
                    use_pi = bool(selected_is_pi[row_index])
                    branch_values = pi if use_pi else p0
                    financing_selected = financingi[row_index] if use_pi else financing0[row_index]
                    equity_selected = equityi[row_index] if use_pi else equity0[row_index]
                    investment_selected = float(i_value) if use_pi else 0.0
                    selected_values = [
                        t_selected[row_index], financing_selected, equity_selected,
                        branch_values["continuation"][row_index], investment_selected,
                    ]
                    rows.append({
                        "episode": episode,
                        "eta": float(eta),
                        "b": float(b_value),
                        "z": float(z_value),
                        "i_index": i_index,
                        "i_value": float(i_value),
                        "production": p0["production"][row_index],
                        "T0": p0["value"][row_index],
                        "TI": pi["value"][row_index],
                        "bp0_star": p0["bp_star"][row_index],
                        "bpI_star": pi["bp_star"][row_index],
                        "q_current_claim": p0["q_current"][row_index],
                        "q_issue_p0": p0["q_issue"][row_index],
                        "q_issue_pi": pi["q_issue"][row_index],
                        "financing_p0": financing0[row_index],
                        "financing_pi": financingi[row_index],
                        "equity_cost_p0": equity0[row_index],
                        "equity_cost_pi": equityi[row_index],
                        "continuation_p0": p0["continuation"][row_index],
                        "continuation_pi": pi["continuation"][row_index],
                        "continuation_source_p0": p0["continuation_source"][row_index],
                        "continuation_source_pi": pi["continuation_source"][row_index],
                        "raw_expected_MP_for_PI": pi["continuation"][row_index] / pi_loss.g,
                        "p_child_p0": p0["p_child"][row_index],
                        "p_child_pi": pi["p_child"][row_index],
                        "default_at_star_p0": p0["default_at_star"][row_index],
                        "default_at_star_pi": pi["default_at_star"][row_index],
                        "selected_branch": "pi" if use_pi else "p0",
                        "selected_is_pi": int(use_pi),
                        "T_selected": t_selected[row_index],
                        "financing_selected": financing_selected,
                        "equity_cost_selected": equity_selected,
                        "continuation_selected": branch_values["continuation"][row_index],
                        "investment_selected": investment_selected,
                        "bp_selected": branch_values["bp_star"][row_index],
                        "q_issue_selected": branch_values["q_issue"][row_index],
                        "T0_identity_error": p0["value"][row_index] - t0_reconstructed[row_index],
                        "TI_identity_error": pi["value"][row_index] - ti_reconstructed[row_index],
                        "T0_finite": bool(np.isfinite(p0["value"][row_index])),
                        "TI_finite": bool(np.isfinite(pi["value"][row_index])),
                        "continuation0_finite": bool(np.isfinite(p0["continuation"][row_index])),
                        "continuationI_finite": bool(np.isfinite(pi["continuation"][row_index])),
                        "q_issue0_finite": bool(np.isfinite(p0["q_issue"][row_index])),
                        "q_issueI_finite": bool(np.isfinite(pi["q_issue"][row_index])),
                        "selected_finite": bool(np.isfinite(selected_values).all()),
                        "Phat_pred": prediction[float(eta)]["Phat"][row_index],
                        "P_pred": prediction[float(eta)]["P"][row_index],
                        "bar_z_pred": prediction[float(eta)]["bar_z"][row_index],
                    })
            print(f"[ep{episode}] teacher i {i_index + 1}/{i_grid_size} ...", flush=True)
    teacher_seconds = time.perf_counter() - teacher_start

    aggregation_start = time.perf_counter()
    by_i = pd.DataFrame(rows)
    selected = aggregate_selected_components(by_i)
    finite_audit = {
        "episode": episode,
        "T0_finite_share": _finite_share(by_i["T0"]),
        "TI_finite_share": _finite_share(by_i["TI"]),
        "continuation0_finite_share": _finite_share(by_i["continuation_p0"]),
        "continuationI_finite_share": _finite_share(by_i["continuation_pi"]),
        "q_issue0_finite_share": _finite_share(by_i["q_issue_p0"]),
        "q_issueI_finite_share": _finite_share(by_i["q_issue_pi"]),
        "selected_finite_share": float(by_i["selected_finite"].mean()),
        "T0_identity_abs_max": float(np.nanmax(np.abs(by_i["T0_identity_error"]))),
        "TI_identity_abs_max": float(np.nanmax(np.abs(by_i["TI_identity_error"]))),
        "Phat_identity_abs_max": float(np.nanmax(np.abs(selected["phat_identity_error"]))),
    }
    for column in ("T0_identity_error", "TI_identity_error"):
        bad = by_i[column].abs() > IDENTITY_TOL
        for record in by_i.loc[bad, ["episode", "eta", "b", "z", "i_index", column]].to_dict("records"):
            errors.append({"block": column, **record})
    bad_phat = selected["phat_identity_error"].abs() > IDENTITY_TOL
    for record in selected.loc[bad_phat, ["episode", "eta", "b", "z", "phat_identity_error"]].to_dict("records"):
        errors.append({"block": "phat_identity_error", **record})
    eta0 = by_i["eta"] == 0.0
    bad_eta0 = eta0 & (
        (by_i["financing_p0"].abs() > ETA_ZERO_TOL)
        | (by_i["financing_pi"].abs() > ETA_ZERO_TOL)
    )
    if bool(bad_eta0.any()):
        warnings.append("eta=0 financing is not numerically zero")
        for record in by_i.loc[bad_eta0, ["episode", "b", "z", "i_index", "financing_p0", "financing_pi"]].to_dict("records"):
            errors.append({"block": "eta0_financing", **record})

    policy_hash_after = model_state_hash(model)
    sdf_hash_after = model_state_hash(sdf_fc1)
    mutation = policy_hash_before != policy_hash_after or sdf_hash_before != sdf_hash_after
    if mutation:
        errors.append({"episode": episode, "block": "model_mutation", "error": "model state hash changed"})
    aggregation_seconds = time.perf_counter() - aggregation_start
    total_seconds = time.perf_counter() - started
    metadata = {
        "episode": episode,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": loaded.metadata.get("checkpoint_sha256"),
        "status": "error_model_mutation" if mutation else "ok",
        "policy_value_hash_before": policy_hash_before,
        "policy_value_hash_after": policy_hash_after,
        "sdf_fc1_hash_before": sdf_hash_before,
        "sdf_fc1_hash_after": sdf_hash_after,
        "i_threshold": float(model.i_threshold),
        "i_grid_size": i_grid_size,
        "shock_seed": shock_seed,
        "n_child_shocks": n_child_shocks,
        "shock_bank_sha256": transition.metadata.get("shock_bank_sha256"),
        "transition_metadata": transition.metadata,
        "timing": {
            "load_seconds": load_seconds,
            "transition_seconds": transition_seconds,
            "teacher_seconds": teacher_seconds,
            "aggregation_seconds": aggregation_seconds,
            "total_seconds": total_seconds,
        },
        "warnings": warnings,
    }
    print(f"[ep{episode}] decomposition written ...", flush=True)
    print(f"[ep{episode}] elapsed {total_seconds:.2f} sec", flush=True)
    return by_i, selected, finite_audit, metadata, errors


def build_eta_contrast(drift: pd.DataFrame) -> pd.DataFrame:
    if drift.empty:
        return pd.DataFrame()
    columns = [
        "episode_from", "episode_to", "b", "z",
        "delta_phat_teacher", "delta_continuation", "delta_financing",
    ]
    eta0 = drift.loc[np.isclose(drift["eta"], 0.0), columns].copy()
    eta1 = drift.loc[np.isclose(drift["eta"], 1.0), columns].copy()
    merged = eta0.merge(
        eta1, on=["episode_from", "episode_to", "b", "z"],
        suffixes=("_eta0", "_eta1"),
    )
    if not merged.empty:
        merged["eta1_minus_eta0_phat_drift"] = (
            merged["delta_phat_teacher_eta1"] - merged["delta_phat_teacher_eta0"]
        )
    return merged


def _plot_line(frame: pd.DataFrame, column: str, output: Path, ylabel: str) -> None:
    subset = frame.loc[np.isclose(frame["b"], 0.05) & np.isclose(frame["eta"], 1.0)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for z_value, group in subset.groupby("z"):
        group = group.sort_values("episode")
        ax.plot(group["episode"], group[column], marker="o", label=f"z={z_value:g}")
    ax.set_xlabel("episode")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_figures(levels: pd.DataFrame, cumulative: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _plot_line(levels, "Phat_teacher", output / "b005_phat_teacher_by_episode.png", "Phat teacher")
    _plot_line(levels, "financing_component", output / "b005_financing_component_by_episode.png", "financing component")
    _plot_line(levels, "continuation_component", output / "b005_continuation_component_by_episode.png", "continuation component")
    _plot_line(levels, "pi_selected_share", output / "b005_pi_selected_share_by_episode.png", "PI selected share")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if cumulative.empty:
        ax.text(
            0.5, 0.5, "EP1 and EP4 are required for cumulative attribution",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output / "b005_ep1_ep4_cumulative_contributions.png", dpi=160)
        plt.close(fig)
        return
    subset = cumulative.loc[
        np.isclose(cumulative["b"], 0.05)
        & np.isclose(cumulative["eta"], 1.0)
        & cumulative["z"].isin([2.0, 3.0, 4.0])
    ]
    components = [
        ("cumulative_financing_contribution", "financing"),
        ("cumulative_continuation_contribution", "continuation"),
        ("cumulative_investment_contribution", "investment"),
        ("cumulative_equity_cost_contribution", "equity cost"),
    ]
    x = np.arange(len(subset))
    width = 0.18
    for index, (column, label) in enumerate(components):
        ax.bar(x + (index - 1.5) * width, subset[column], width=width, label=label)
    ax.set_xticks(x, [f"z={value:g}" for value in subset.get("z", [])])
    ax.set_ylabel("EP1 to EP4 contribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "b005_ep1_ep4_cumulative_contributions.png", dpi=160)
    plt.close(fig)


def summary_markdown(
    levels: pd.DataFrame,
    cumulative: pd.DataFrame,
    finite: pd.DataFrame,
    errors: pd.DataFrame,
) -> str:
    lines = ["# P-teacher drift summary", ""]
    finite_ok = not finite.empty and bool((finite.filter(like="finite_share") == 1.0).all().all())
    identity_columns = [column for column in finite.columns if "identity_abs_max" in column]
    identity_ok = not finite.empty and all(
        float(finite[column].max()) <= IDENTITY_TOL for column in identity_columns
    )
    lines.extend([
        f"1. 数据完整性：完成 episodes = {sorted(levels['episode'].unique().tolist()) if not levels.empty else []}",
        f"2. finite audit：{'PASS' if finite_ok else 'INCOMPLETE/FAIL'}",
        f"3. identity check：{'PASS' if identity_ok else 'FAIL'}",
    ])
    if cumulative.empty:
        lines.extend([
            "4. EP1 -> EP4 upward drift：不可判断，缺少 EP1 或 EP4。",
            "5. b=.05,z=2,eta=1：不可判断。",
            "6. b=.05,z=3,eta=1：不可判断。",
            "7. b=.05,z=4,eta=1：不可判断。",
        ])
        focus = pd.DataFrame()
    else:
        focus = cumulative.loc[
            np.isclose(cumulative["b"], 0.05)
            & np.isclose(cumulative["eta"], 1.0)
            & cumulative["z"].isin([2.0, 3.0, 4.0])
        ]
        upward = bool((focus["cumulative_phat_drift"] > 0).any()) if not focus.empty else False
        lines.append(f"4. EP1 -> EP4 upward drift：{'存在' if upward else '未在重点状态发现'}")
        for number, z_value in zip((5, 6, 7), (2.0, 3.0, 4.0)):
            row = focus.loc[np.isclose(focus["z"], z_value)]
            if row.empty:
                lines.append(f"{number}. b=.05,z={z_value:g},eta=1：缺失。")
            else:
                item = row.iloc[0]
                lines.append(
                    f"{number}. b=.05,z={z_value:g},eta=1：ΔPhat={item['cumulative_phat_drift']:.6g}, "
                    f"financing={item['cumulative_financing_contribution']:.6g}, "
                    f"continuation={item['cumulative_continuation_contribution']:.6g}, "
                    f"label={item['mechanism_label']}。"
                )
    for number, column, label in (
        (8, "cumulative_financing_contribution", "financing contribution"),
        (9, "cumulative_continuation_contribution", "continuation contribution"),
        (10, "cumulative_investment_contribution", "investment contribution"),
        (11, "cumulative_equity_cost_contribution", "equity financing cost contribution"),
        (12, "cumulative_pi_selected_share_change", "PI-selected-share change"),
    ):
        value = float(focus[column].mean()) if not focus.empty else float("nan")
        lines.append(f"{number}. {label}：{value:.6g}" if np.isfinite(value) else f"{number}. {label}：不可判断。")
    eta0 = cumulative.loc[np.isclose(cumulative["eta"], 0.0)] if not cumulative.empty else cumulative
    eta0_financing = float(eta0["cumulative_financing_contribution"].abs().max()) if not eta0.empty else float("nan")
    lines.append(
        f"13. eta0 control：max |financing drift|={eta0_financing:.6g}。"
        if np.isfinite(eta0_financing) else "13. eta0 control：不可判断。"
    )
    gap = float(levels["phat_pred_minus_teacher"].abs().mean()) if not levels.empty else float("nan")
    lines.append(f"14. Phat_pred vs Phat_teacher gap：mean abs gap={gap:.6g}。" if np.isfinite(gap) else "14. Phat gap：不可判断。")
    labels = sorted(focus["mechanism_label"].unique().tolist()) if not focus.empty else []
    if finite_ok and identity_ok and errors.empty:
        lines.append(f"15. 当前最支持的机制：{', '.join(labels) if labels else '未识别'}。")
    else:
        lines.append("15. 当前最支持的机制：因 nonfinite 或 identity failure，不给出强机制结论。")
    lines.append("16. 未解决问题：这是固定状态、固定 shock 的描述性 operator decomposition，不是 causal DiD，也不包含 lagged-teacher audit。")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--episodes", default="0:4")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--b-values", type=float, nargs="+", default=list(DEFAULT_B_VALUES))
    parser.add_argument("--z-values", type=float, nargs="+", default=list(DEFAULT_Z_VALUES))
    parser.add_argument("--eta-values", type=float, nargs="+", default=list(DEFAULT_ETA_VALUES))
    parser.add_argument("--n-child-shocks", type=int, default=64)
    parser.add_argument("--shock-seed", type=int, default=12345)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty; use --overwrite: {output}")
    for directory in (output / "episodes", output / "tables", output / "figures"):
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(device)
    if args.n_child_shocks < 2:
        raise SystemExit("--n-child-shocks must be at least 2")
    if any(value not in (0.0, 1.0) for value in args.eta_values):
        raise SystemExit("--eta-values currently supports only 0 and 1")

    discovered = discover_checkpoints(run_root)
    requested = parse_episode_selection(args.episodes)
    episodes = sorted(discovered) if requested is None else sorted(requested & set(discovered))
    missing = sorted((requested or set()) - set(discovered))
    if missing:
        raise SystemExit(f"missing combined checkpoints for episodes: {missing}")
    if not episodes:
        raise SystemExit(f"no selected combined checkpoints under {run_root}")

    reference_firm, reference_macro = choose_reference_artifacts(run_root)
    _reference_frame, reference = load_reference_state(reference_firm, macro_path=reference_macro)
    all_by_i: List[pd.DataFrame] = []
    all_selected: List[pd.DataFrame] = []
    finite_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    first_model_spec: Dict[str, Any] | None = None
    expected_shock_hash: str | None = None

    for episode in episodes:
        episode_dir = output / "episodes" / f"ep_{episode:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        try:
            by_i, selected, finite, metadata, errors = evaluate_episode(
                episode=episode,
                checkpoint=discovered[episode],
                reference=reference,
                b_values=args.b_values,
                z_values=args.z_values,
                eta_values=args.eta_values,
                n_child_shocks=args.n_child_shocks,
                shock_seed=args.shock_seed,
                device=device,
            )
            spec = {
                "i_threshold": metadata["i_threshold"],
                "i_grid_size": metadata["i_grid_size"],
            }
            if first_model_spec is None:
                first_model_spec = spec
                atomic_write_json(output / "canonical_reference.json", {
                    "x": reference.x,
                    "Hatcf": reference.hatcf,
                    "LnKF": reference.lnkf,
                    "hatc_cal": reference.hatc_cal,
                    "lnk_cal": reference.lnk_cal,
                    "i_threshold": spec["i_threshold"],
                    "i_grid_size": spec["i_grid_size"],
                    "reference_firm_data": str(reference_firm),
                    "reference_macro_data": reference_macro,
                })
            elif spec != first_model_spec:
                metadata["warnings"].append(
                    f"model i-grid differs from episode {episodes[0]}: {spec} != {first_model_spec}"
                )
            shock_hash = metadata["shock_bank_sha256"]
            if expected_shock_hash is None:
                expected_shock_hash = shock_hash
            elif shock_hash != expected_shock_hash:
                metadata["status"] = "error_shock_bank_mismatch"
                errors.append({
                    "episode": episode,
                    "block": "shock_bank",
                    "error": f"{shock_hash} != {expected_shock_hash}",
                })

            atomic_write_json(episode_dir / "metadata.json", metadata)
            atomic_write_json(episode_dir / "finite_audit.json", finite)
            atomic_write_csv(episode_dir / "p_teacher_components_by_i.csv", by_i)
            atomic_write_csv(episode_dir / "p_teacher_selected_branch.csv", selected)
            error_rows.extend(errors)
            if metadata["status"] == "ok":
                all_by_i.append(by_i)
                all_selected.append(selected)
                finite_rows.append(finite)
            else:
                print(f"[ep{episode}] HARD FAIL: {metadata['status']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            error = {
                "episode": episode,
                "block": "episode",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=10),
            }
            error_rows.append(error)
            atomic_write_json(episode_dir / "metadata.json", {"episode": episode, "status": "error", **error})
            print(f"[ep{episode}] FAILED: {error['error']}", flush=True)

        merged_by_i = pd.concat(all_by_i, ignore_index=True) if all_by_i else pd.DataFrame()
        merged_selected = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
        finite_frame = pd.DataFrame(finite_rows)
        errors_frame = pd.DataFrame(error_rows)
        if errors_frame.empty:
            errors_frame = pd.DataFrame(columns=ERROR_COLUMNS)
        atomic_write_csv(output / "tables" / "p_teacher_components_by_i.csv", merged_by_i)
        atomic_write_csv(output / "tables" / "p_teacher_selected_branch.csv", merged_selected)
        atomic_write_csv(output / "tables" / "finite_audit.csv", finite_frame)
        atomic_write_csv(output / "errors.csv", errors_frame)
        if not merged_selected.empty:
            drift = compute_episode_drift(merged_selected)
            cumulative = compute_cumulative_drift(merged_selected)
            eta_contrast = build_eta_contrast(drift)
            focus = merged_selected.loc[np.isclose(merged_selected["b"], 0.05)]
            focus_drift = drift.loc[np.isclose(drift["b"], 0.05)] if not drift.empty else drift
            atomic_write_csv(output / "tables" / "p_teacher_drift.csv", drift)
            atomic_write_csv(output / "tables" / "p_teacher_cumulative_ep1_ep4.csv", cumulative)
            atomic_write_csv(output / "tables" / "p_teacher_eta_contrast.csv", eta_contrast)
            atomic_write_csv(output / "tables" / "p_teacher_focus_b005.csv", focus)
            atomic_write_csv(output / "tables" / "p_teacher_focus_b005_drift.csv", focus_drift)

    merged_selected = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    finite_frame = pd.DataFrame(finite_rows)
    errors_frame = pd.DataFrame(error_rows)
    if errors_frame.empty:
        errors_frame = pd.DataFrame(columns=ERROR_COLUMNS)
    drift = compute_episode_drift(merged_selected) if not merged_selected.empty else pd.DataFrame()
    cumulative = compute_cumulative_drift(merged_selected) if not merged_selected.empty else pd.DataFrame()
    write_figures(merged_selected, cumulative, output / "figures")
    (output / "p_teacher_drift_summary.md").write_text(
        summary_markdown(merged_selected, cumulative, finite_frame, errors_frame),
        encoding="utf-8",
    )
    atomic_write_json(output / "metadata.json", {
        "run_root": run_root,
        "episodes_requested": args.episodes,
        "episodes_completed": sorted(merged_selected["episode"].unique().tolist()) if not merged_selected.empty else [],
        "b_values": args.b_values,
        "z_values": args.z_values,
        "eta_values": args.eta_values,
        "focus_parent_states_per_i": len(args.b_values) * len(args.z_values) * len(args.eta_values),
        "n_child_shocks": args.n_child_shocks,
        "shock_seed": args.shock_seed,
        "shock_bank_sha256": expected_shock_hash,
        "read_only": True,
        "large_orchestration_called": False,
    })


if __name__ == "__main__":
    main()
