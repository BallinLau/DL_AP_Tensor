"""Firm Bellman banks extracted from root-cleared full-economy transitions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from training.inner_fixed_point import InnerFPChildBank, InnerFPCollocationBank


PARENT_COLUMNS = ("b", "z", "ETA", "i", "x", "Hatcf", "LnKF")
CHILD_COLUMNS = ("b", "z", "ETA", "i", "x", "Hatcf", "LnKF", "M")
SHOCK_COLUMNS = (
    "eps_x_transition",
    "eps_z_transition",
    "u_eta_transition",
    "u_i_transition",
)


def _frame(table: Any) -> pd.DataFrame:
    frame = table.to_dataframe()
    for name in ("path", "t", "branch", "ID"):
        if name in frame:
            frame[name] = np.rint(frame[name]).astype(np.int64)
    return frame


def _require_root_cleared_macro(
    macro: pd.DataFrame,
    tolerance: float,
    *,
    solver_mode: str = "strict_global",
) -> dict[str, Any]:
    required = {
        "market_root_found",
        "market_committed_residual_normalized",
        "Hatc_cleared",
        "Hatc_firm_input",
    }
    missing = sorted(required - set(macro.columns))
    if missing:
        raise RuntimeError(f"root-cleared macro table is missing {missing}")
    residual = macro["market_committed_residual_normalized"].abs()
    strict = (
        (macro["market_root_found"] == 1.0)
        & np.isfinite(residual)
        & (residual <= float(tolerance))
    )
    approximate = pd.Series(False, index=macro.index, dtype=bool)
    if solver_mode == "fc1_bounded_improvement_pilot":
        approximate_columns = {
            "fast_solver_status",
            "corrected_normalized_residual",
            "improvement_fraction",
            "h_policy_minus_h_cal",
            "resource_feasible",
        }
        approximate_missing = sorted(approximate_columns - set(macro.columns))
        if approximate_missing:
            raise RuntimeError(
                "bounded-clearing macro table is missing "
                f"{approximate_missing}"
            )
        corrected = macro["corrected_normalized_residual"].abs()
        mismatch = macro["h_policy_minus_h_cal"].abs()
        improvement = macro["improvement_fraction"]
        approximate = (
            (macro["fast_solver_status"] == 1.0)
            & (macro["resource_feasible"] == 1.0)
            & np.isfinite(corrected)
            & np.isfinite(mismatch)
            & np.isfinite(improvement)
            & (corrected <= 1.0e-3)
            & (mismatch <= 0.005)
            & (improvement >= 0.90)
        )
    accepted = strict | approximate
    if not accepted.all():
        columns = ["path", "t", "branch", "market_root_found"]
        for name in (
            "fast_solver_status",
            "corrected_normalized_residual",
            "h_policy_minus_h_cal",
            "improvement_fraction",
        ):
            if name in macro.columns:
                columns.append(name)
        bad = macro.loc[~accepted, columns]
        raise RuntimeError(
            "Firm transition bank contains a node outside the declared strict/"
            "approximate clearing gates: "
            f"{bad.head(20).to_dict(orient='records')}"
        )
    if not np.allclose(
        macro["Hatc_cleared"], macro["Hatc_firm_input"], atol=0.0, rtol=0.0
    ):
        raise RuntimeError("cleared Hatc was not the Firm input at every node")
    return {
        "solver_mode": str(solver_mode),
        "node_count": int(len(macro)),
        "strict_node_count": int(strict.sum()),
        "approximate_node_count": int(approximate.sum()),
        "strict_clearing_share": float(strict.mean()),
        "approximate_clearing_share": float(approximate.mean()),
        "strict_tolerance": float(tolerance),
        "approximate_normalized_residual_tolerance": 1.0e-3,
        "approximate_log_mismatch_tolerance": 0.005,
        "approximate_minimum_improvement": 0.90,
    }


def build_market_cleared_transition_bank(
    output: Any,
    *,
    n_parents: int,
    seed: int,
    split: str,
    root_tolerance: float,
    solver_mode: str = "strict_global",
) -> tuple[InnerFPCollocationBank, InnerFPChildBank, dict[str, Any]]:
    """Match each incumbent parent firm to both independently-cleared children."""
    firm = _frame(output.firm)
    macro = _frame(output.macro)
    clearing_audit = _require_root_cleared_macro(
        macro, root_tolerance, solver_mode=solver_mode
    )
    if "eps_x_transition" not in firm.columns:
        if "eps_x_transition" not in macro.columns:
            raise RuntimeError("root-cleared output lacks macro eps_x_transition")
        firm = firm.merge(
            macro.loc[:, ["path", "t", "branch", "eps_x_transition"]],
            on=["path", "t", "branch"],
            how="left",
            validate="many_to_one",
        )
    missing_firm = sorted(
        set(PARENT_COLUMNS + CHILD_COLUMNS + SHOCK_COLUMNS) - set(firm.columns)
    )
    if missing_firm:
        raise RuntimeError(f"Firm transition table is missing {missing_firm}")

    parent = firm.loc[firm["branch"] == -1].copy()
    child = firm.loc[firm["branch"].isin((0, 1))].copy()
    parent["child_t"] = parent["t"] + 1
    key = ["path", "child_t", "ID"]
    child = child.rename(columns={"t": "child_t"})
    child0 = child.loc[child["branch"] == 0].copy()
    child1 = child.loc[child["branch"] == 1].copy()
    paired = parent.merge(
        child0,
        on=key,
        how="inner",
        suffixes=("_parent", "_child0"),
        validate="one_to_one",
    ).merge(
        child1,
        on=key,
        how="inner",
        suffixes=("", "_child1"),
        validate="one_to_one",
    )
    # The second merge leaves branch-0 names with _child0 and branch-1 names
    # either unsuffixed or _child1 depending on collision. Resolve explicitly.
    def column(base: str, branch: int) -> str:
        candidates = (
            (f"{base}_child0", f"{base}_x", base)
            if branch == 0
            else (f"{base}_child1", f"{base}_y", base)
        )
        for candidate in candidates:
            if candidate in paired.columns:
                return candidate
        raise RuntimeError(f"cannot resolve child-{branch} column {base!r}")

    def parent_column(base: str) -> str:
        for candidate in (f"{base}_parent", base):
            if candidate in paired.columns:
                return candidate
        raise RuntimeError(f"cannot resolve parent column {base!r}")

    if paired.empty:
        raise RuntimeError("root-cleared rollout produced no matched Firm transitions")
    aggregation_mode = str(
        getattr(output, "meta", {}).get("aggregation", {}).get(
            "mode", "legacy_mixed_replay"
        )
    )
    mass_column = parent_column("firm_mass") if "firm_mass" in firm.columns else None
    finite_columns = [parent_column(name) for name in PARENT_COLUMNS]
    for branch in (0, 1):
        finite_columns.extend(column(name, branch) for name in CHILD_COLUMNS)
        finite_columns.extend(column(name, branch) for name in SHOCK_COLUMNS)
    finite = np.isfinite(
        paired.loc[:, finite_columns].to_numpy(dtype=np.float64)
    ).all(axis=1)
    paired = paired.loc[finite].reset_index(drop=True)
    if paired.empty:
        raise RuntimeError("all matched root-cleared Firm transitions are non-finite")
    available = int(len(paired))
    if len(paired) < int(n_parents):
        raise RuntimeError(
            "insufficient natural matched Firm transitions: "
            f"available={len(paired)}, requested={n_parents}"
        )
    rng = np.random.default_rng(int(seed))
    probabilities = None
    if aggregation_mode == "continuum_mass_consistent_v1":
        if mass_column is None:
            raise RuntimeError("continuum Firm bank is missing firm_mass")
        mass = paired[mass_column].to_numpy(dtype=np.float64)
        if not np.isfinite(mass).all() or (mass <= 0.0).any():
            raise RuntimeError("continuum Firm bank contains invalid mass")
        probabilities = mass / mass.sum()
    chosen = rng.choice(
        len(paired), size=int(n_parents), replace=False, p=probabilities
    )
    paired = paired.iloc[chosen].reset_index(drop=True)

    parent_tensor = torch.as_tensor(
        paired.loc[:, [parent_column(name) for name in PARENT_COLUMNS]].to_numpy(
            dtype=np.float32
        )
    )
    children = []
    shocks: dict[str, list[torch.Tensor]] = {name: [] for name in SHOCK_COLUMNS}
    for branch in (0, 1):
        children.append(
            torch.as_tensor(
                paired.loc[:, [column(name, branch) for name in CHILD_COLUMNS]].to_numpy(
                    dtype=np.float32
                )
            )
        )
        for name in SHOCK_COLUMNS:
            shocks[name].append(
                torch.as_tensor(
                    paired.loc[:, column(name, branch)].to_numpy(dtype=np.float32)
                ).reshape(-1, 1)
            )
    child_tensor = torch.stack(children, dim=1)
    shock_tensor = {
        name: torch.stack(values, dim=1) for name, values in shocks.items()
    }
    parent_bank = InnerFPCollocationBank(
        parent_states=parent_tensor,
        source_kind=torch.zeros(len(parent_tensor), dtype=torch.long),
        seed=int(seed),
        split=str(split),
        metadata={
            "definition": "natural_declared_gate_cleared_full_economy_parent_firms",
            "state_columns": list(PARENT_COLUMNS),
            "sampling": "without_replacement",
            "available_matched_transitions": available,
            "aggregation_mode": aggregation_mode,
            "sampling_mass_weighted": probabilities is not None,
        },
    )
    child_bank = InnerFPChildBank(
        children=child_tensor,
        eps_x=shock_tensor["eps_x_transition"],
        eps_z=shock_tensor["eps_z_transition"],
        u_eta=shock_tensor["u_eta_transition"],
        u_i=shock_tensor["u_i_transition"],
        seed=int(seed),
        metadata={
            "definition": "two_independently_declared_gate_cleared_production_children",
            "continuous_child_shocks": 2,
            "parent_fingerprint": parent_bank.fingerprint,
            "root_tolerance": float(root_tolerance),
            "artificial_b_only_used": False,
        },
    )
    audit = {
        "split": str(split),
        "available_matched_transitions": available,
        "selected_transitions": int(n_parents),
        "parent_fingerprint": parent_bank.fingerprint,
        "child_fingerprint": child_bank.fingerprint,
        "child_branches": [0, 1],
        "root_tolerance": float(root_tolerance),
        "artificial_b_only_used": False,
        "aggregation_mode": aggregation_mode,
        "sampling_mass_weighted": probabilities is not None,
        "clearing_gate": clearing_audit,
    }
    return parent_bank, child_bank, audit
