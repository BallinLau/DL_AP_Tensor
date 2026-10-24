"""Fixed evaluation banks (``evaluation_bank_v1``) for the equilibrium convergence evaluator.

Design (Phase A §3): a bank stores **parent states + fixed exogenous shocks +
macro context** — never precomputed children.  Children (including
``hatcf_next``/``lnkf_next`` and the SDF stochastic discount factor ``M``) are
rebuilt *per checkpoint* by that checkpoint's own FC1/SDF under the fixed
shocks via :func:`build_transition_for_checkpoint`, so that one bank is shared
by every episode while each checkpoint's own beliefs still drive its children.

Bank overview:

* **A** fixed natural firm test — branch-filtered parent rows from a firm/macro
  panel (natural eta distribution, source_id / lineage_id / pairing keys kept,
  independent sampling seed).  When the panel comes from a baseline-checkpoint
  simulation the bank is tagged ``reference-policy-distribution``.
* **B** balanced eta — the same parent states as two counterfactual views with
  the ETA column covered by eta=0 / eta=1 (``counterfactual-eta-view``, equal
  view weights).  Child rebuilding uses the parent-view eta financing gate.
* **C** boundary/stress — a fixed synthetic stress grid (b endpoints and high
  leverage, z tails in [-2.5, 2.5], i in {0.10, 0.25, 0.45}, eta in {0, 1},
  three macro anchors Recession/Normal/Expansion taken as *joint* samples from
  Bank D conditional x quantiles).  Tagged ``out-of-distribution-stress``.
* **D** macro holdout — independent-path macro parent-child rows paired **only
  by key joins** (``(path, t, branch, economic_node_id)``), never by row
  adjacency.
* **E** fixed rollout — fixed initial cross-section ``{b0, z0, eta0, i0, K0,
  x0}`` + a fixed shock specification following the ``stable_firm_identity``
  keyed-RNG convention of ``data/simulate_ts_parallel`` (key = (seed, path,
  economic_time, branch, identity, shock_type)) + Low/Baseline/High initial
  leverage variants with comparable capital totals.

Persistence: :func:`save_bank` writes ``bank_manifest.json`` + one ``.pt`` per
bank and refuses to overwrite an existing root (an explicit ``version`` upgrade
writes the new version and returns an explicit *incomparable* warning).
:func:`load_bank` re-verifies every file sha256.

Everything here is CPU-only pure functions + dataclasses.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from analysis.convergence_transition import (
    ConvergenceShockBank,
    FirmStateIndex,
    MacroTransitionContext,
    build_child_exogenous_bundle,
)
from analysis.economic_config import AnalysisEconomicConfig
from config import HyperParams
from utils.firm_transition import apply_refinancing_policy, expand_children_exact_eta_tensor


DEFAULT_BANK_VERSION = "evaluation_bank_v1"
BANK_MANIFEST_FILENAME = "bank_manifest.json"
BANK_SCHEMA_VERSION = "evaluation_bank_v1/bank"
MANIFEST_SCHEMA_VERSION = "evaluation_bank_v1/manifest"

BANK_KEYS = ("A", "B", "C", "D", "E")

# firm_state column order: [b, z, ETA, i, x, Hatcf, LnKF] (must not be changed)
STATE_COLUMNS = ("b", "z", "ETA", "i", "x", "Hatcf", "LnKF")
PAIRING_KEY_COLUMNS = ("path", "ID", "t", "branch", "economic_node_id")

SOURCE_ID_SIMULATED = 0
SOURCE_ID_COVERAGE = 1
SOURCE_ID_UNKNOWN = -1

TAG_REFERENCE_POLICY_DISTRIBUTION = "reference-policy-distribution"
TAG_COUNTERFACTUAL_ETA_VIEW = "counterfactual-eta-view"
TAG_OUT_OF_DISTRIBUTION_STRESS = "out-of-distribution-stress"

DEFAULT_LEVERAGE_VARIANTS = ("Low", "Baseline", "High")

REASON_BANK_ROOT_EXISTS = "BANK_ROOT_EXISTS"
REASON_BANK_VERSION_NOT_UPGRADED = "BANK_VERSION_NOT_UPGRADED"
REASON_BANK_FILE_HASH_MISMATCH = "BANK_FILE_HASH_MISMATCH"
REASON_MISSING_BANK_MANIFEST = "MISSING_BANK_MANIFEST"
REASON_MISSING_BRANCH_COLUMN = "MISSING_BRANCH_COLUMN"
REASON_NO_PARENT_VIEW_ROWS = "NO_PARENT_VIEW_ROWS"
REASON_MACRO_JOIN_UNMATCHED = "MACRO_JOIN_UNMATCHED"
REASON_MACRO_DUPLICATE_KEY = "MACRO_DUPLICATE_KEY"
REASON_DUPLICATE_CHILD_KEY = "DUPLICATE_CHILD_KEY"
REASON_MACRO_CONTEXT_MISSING = "MACRO_CONTEXT_MISSING"


class EvalBankError(ValueError):
    """Evaluation-bank error carrying a stable machine reason code."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(f"[{reason_code}] {message}")
        self.reason_code = reason_code
        self.message = message


# ---------------------------------------------------------------------------
# hashing helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_value(digest: "hashlib._Hash", value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor:")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"map:")
        for key in sorted(value.keys()):
            digest.update(str(key).encode("utf-8"))
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"seq:")
        for item in value:
            _hash_value(digest, item)
    else:
        digest.update(b"val:")
        digest.update(repr(value).encode("utf-8"))


def content_sha256(payload: Any) -> str:
    """Canonical content hash of a bank payload (independent of file encoding)."""
    digest = hashlib.sha256()
    _hash_value(digest, payload)
    return digest.hexdigest()


def shock_bank_sha256(bank: ConvergenceShockBank) -> str:
    """Stable hash of a shock bank's tensor content (see bp_diagnostics)."""
    digest = hashlib.sha256()
    for name in ("eps_x", "eps_z", "u_eta", "u_i"):
        value = getattr(bank, name).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def compute_config_snapshot_hash(
    config: AnalysisEconomicConfig | Mapping[str, Any],
) -> str:
    values = config.to_dict() if isinstance(config, AnalysisEconomicConfig) else dict(config)
    canonical = json.dumps(values, sort_keys=True, default=float)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evaluation_code_commit() -> Optional[str]:
    """Git commit of the checkout that produced the bank (None if unavailable)."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        output = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    commit = (output.stdout or "").strip()
    return commit or None


def lineage_id_for(path: Any, firm_id: Any) -> str:
    """Lineage identity derived from (path, ID)."""
    return f"{path}::{firm_id}"


def hash_split_lineages(
    lineage_ids: Iterable[str],
    *,
    holdout_fraction: float,
    salt: str = "evaluation_bank_v1/train-split",
) -> Dict[str, List[str]]:
    """Deterministic path/lineage hash split for training-data leakage control.

    ``holdout`` lineages stay in the evaluation bank, ``excluded`` lineages are
    the training side and must not be evaluated on.  Splitting is a pure hash
    function of ``(salt, lineage_id)`` so it is identical for every episode.
    """
    if not 0.0 <= float(holdout_fraction) <= 1.0:
        raise EvalBankError("INVALID_HOLDOUT_FRACTION", f"holdout_fraction must be in [0,1], got {holdout_fraction}")
    holdout: List[str] = []
    excluded: List[str] = []
    for lineage in sorted({str(item) for item in lineage_ids}):
        digest = hashlib.sha256(f"{salt}:{lineage}".encode("utf-8")).hexdigest()
        unit = int(digest[:8], 16) / float(0xFFFFFFFF)
        if unit < float(holdout_fraction):
            holdout.append(lineage)
        else:
            excluded.append(lineage)
    return {"holdout": holdout, "excluded": excluded, "salt": salt, "holdout_fraction": float(holdout_fraction)}


# ---------------------------------------------------------------------------
# panel extraction (explicit branch filtering + key-based pairing)
# ---------------------------------------------------------------------------


def select_parent_view_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Explicitly filter the parent view of a simulate-style panel.

    The panel exports every main-branch node twice: as a child view
    ``(t, branch=0)`` (pre-exit) and as a parent view ``(t, branch=-1)``
    (post-exit).  Parent selection prefers ``branch == -1`` and falls back to
    ``branch == 0``; anything else would silently double count the dual views.
    """
    if "branch" not in frame.columns:
        raise EvalBankError(
            REASON_MISSING_BRANCH_COLUMN,
            "panel frame is missing the 'branch' column; explicit branch filtering "
            "is mandatory (dual-view nodes would otherwise be double counted)",
        )
    branch = pd.to_numeric(frame["branch"], errors="coerce")
    parents = frame.loc[branch == -1]
    if parents.empty:
        parents = frame.loc[branch == 0]
    if parents.empty:
        raise EvalBankError(
            REASON_NO_PARENT_VIEW_ROWS,
            "panel frame has no parent-view rows (branch == -1 preferred, branch == 0 fallback)",
        )
    return parents.copy()


def attach_macro_context(
    parents: pd.DataFrame,
    macro_frame: pd.DataFrame,
    *,
    macro_keys: Sequence[str] = ("path", "t", "branch"),
) -> pd.DataFrame:
    """Attach the realized macro context ``hatc_cal``/``lnk_cal`` to parent rows.

    The macro panel's ``Hatc``/``LnK`` are *realized aggregation* values while
    ``hatcf``/``lnkf`` are beliefs and the firm panel's ``Hatcf``/``LnKF`` are
    the firm-side belief states.  Only ``Hatc``/``LnK`` may become the transition
    macro context; the join is on explicit keys and never by row adjacency or
    positional de-duplication.
    """
    keys = [key for key in macro_keys if key in macro_frame.columns and key in parents.columns]
    if not keys:
        raise EvalBankError(REASON_MACRO_JOIN_UNMATCHED, "no shared join keys between parents and macro frame")
    if not {"Hatc", "LnK"}.issubset(macro_frame.columns):
        raise EvalBankError(
            REASON_MACRO_CONTEXT_MISSING,
            "macro frame must provide realized aggregate columns 'Hatc' and 'LnK'",
        )
    macro_key_frame = macro_frame[keys + ["Hatc", "LnK"]].copy()
    duplicated = macro_key_frame.duplicated(subset=keys, keep=False)
    if bool(duplicated.any()):
        raise EvalBankError(
            REASON_MACRO_DUPLICATE_KEY,
            f"macro frame has duplicated join keys {list(keys)}; refusing ambiguous join",
        )
    merged = parents.merge(
        macro_key_frame.rename(columns={"Hatc": "hatc_cal", "LnK": "lnk_cal"}),
        on=keys,
        how="left",
        validate="many_to_one",
    )
    unmatched = merged["hatc_cal"].isna()
    if bool(unmatched.any()):
        raise EvalBankError(
            REASON_MACRO_JOIN_UNMATCHED,
            f"{int(unmatched.sum())} parent rows have no macro context on keys {list(keys)}",
        )
    return merged


def pair_firm_parent_child(
    firm_frame: pd.DataFrame,
    *,
    required_branches: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pair parent firm rows with child rows strictly by key.

    Child rows of parent ``(path, ID, t)`` are keyed ``(path, ID, t + 1,
    branch = k)`` for every required branch ``k`` — the same rule as the
    production batcher (``training/episode.py::_create_firm_batches_from_df``).
    Row adjacency is never used: the panel is written node-block-wise, so a
    firm's ``t`` and ``t + 1`` rows are not adjacent.

    Returns ``(parents, pair_table)`` where ``pair_table`` has one row per
    (parent, child-branch) candidate with ``matched`` telling whether the keyed
    child row exists.
    """
    parents = select_parent_view_rows(firm_frame)
    branch_all = pd.to_numeric(firm_frame["branch"], errors="coerce")
    child_rows = firm_frame.loc[branch_all >= 0]
    if required_branches is None:
        branches = sorted({int(value) for value in pd.to_numeric(child_rows["branch"], errors="coerce").dropna()})
    else:
        branches = [int(value) for value in required_branches]
    if not branches:
        raise EvalBankError(
            REASON_NO_PARENT_VIEW_ROWS,
            "panel frame has no child rows (branch >= 0) to pair with parent rows",
        )

    parent_keys = parents[list(PAIRING_KEY_COLUMNS)].copy()
    parent_keys["t"] = pd.to_numeric(parent_keys["t"]).astype(np.int64)
    parent_keys["branch"] = pd.to_numeric(parent_keys["branch"]).astype(np.int64)
    parent_keys["path"] = pd.to_numeric(parent_keys["path"]).astype(np.int64)
    parent_keys["economic_node_id"] = pd.to_numeric(parent_keys["economic_node_id"]).astype(np.int64)
    parent_keys["ID"] = parent_keys["ID"].astype(str)

    child_keys = child_rows[list(PAIRING_KEY_COLUMNS)].copy()
    child_keys["t"] = pd.to_numeric(child_keys["t"]).astype(np.int64)
    child_keys["branch"] = pd.to_numeric(child_keys["branch"]).astype(np.int64)
    child_keys["path"] = pd.to_numeric(child_keys["path"]).astype(np.int64)
    child_keys["economic_node_id"] = pd.to_numeric(child_keys["economic_node_id"]).astype(np.int64)
    child_keys["ID"] = child_keys["ID"].astype(str)

    pair_frames: List[pd.DataFrame] = []
    for child_branch in branches:
        candidates = child_keys.loc[child_keys["branch"] == child_branch]
        duplicates = candidates.duplicated(subset=["path", "ID", "t"], keep=False)
        if bool(duplicates.any()):
            raise EvalBankError(
                REASON_DUPLICATE_CHILD_KEY,
                f"duplicate child rows for (path, ID, t) at branch={child_branch}",
            )
        lookup = candidates.rename(
            columns={"t": "child_t", "economic_node_id": "child_economic_node_id"}
        )[["path", "ID", "child_t", "child_economic_node_id"]]
        left = parent_keys.copy()
        left["child_t"] = left["t"] + 1
        merged = left.merge(lookup, on=["path", "ID", "child_t"], how="left")
        merged["child_branch"] = child_branch
        merged["matched"] = merged["child_economic_node_id"].notna()
        pair_frames.append(merged)

    pair_table = pd.concat(pair_frames, ignore_index=True)
    pair_table = pair_table.sort_values(
        ["path", "ID", "t", "child_branch"], kind="mergesort"
    ).reset_index(drop=True)
    return parents, pair_table


def pair_macro_parent_child(macro_frame: pd.DataFrame) -> pd.DataFrame:
    """Pair macro parent rows with their next-period macro rows by key only.

    Pairs join ``(path, t, branch) -> (path, t + 1, branch)``.  Row adjacency is
    explicitly forbidden (panel row order is node export order, not time order).
    """
    parents = select_parent_view_rows(macro_frame)
    required = {"path", "t", "branch", "economic_node_id", "x", "hatcf", "lnkf", "Hatc", "LnK"}
    missing = sorted(required - set(macro_frame.columns))
    if missing:
        raise EvalBankError(REASON_MACRO_CONTEXT_MISSING, f"macro frame is missing columns: {missing}")

    left = parents.copy()
    left["t"] = pd.to_numeric(left["t"]).astype(np.int64)
    left["branch"] = pd.to_numeric(left["branch"]).astype(np.int64)
    left["path"] = pd.to_numeric(left["path"]).astype(np.int64)
    left = left.rename(
        columns={
            "economic_node_id": "economic_node_id",
            "x": "x_t",
            "hatcf": "hatcf_t",
            "lnkf": "lnkf_t",
            "Hatc": "hatc_t",
            "LnK": "lnk_t",
        }
    )
    left["t1"] = left["t"] + 1

    right = macro_frame.copy()
    right["t"] = pd.to_numeric(right["t"]).astype(np.int64)
    right["branch"] = pd.to_numeric(right["branch"]).astype(np.int64)
    right["path"] = pd.to_numeric(right["path"]).astype(np.int64)
    right = right.rename(
        columns={
            "economic_node_id": "economic_node_id_child",
            "x": "x_t1",
            "Hatc": "hatc_t1",
            "LnK": "lnk_t1",
        }
    )
    duplicates = right.duplicated(subset=["path", "t", "branch"], keep=False)
    if bool(duplicates.any()):
        raise EvalBankError(REASON_MACRO_DUPLICATE_KEY, "macro frame has duplicated (path, t, branch) keys")
    right = right[["path", "t", "branch", "economic_node_id_child", "x_t1", "hatc_t1", "lnk_t1"]]

    merged = left.merge(
        right.rename(columns={"t": "t1"}),
        on=["path", "t1", "branch"],
        how="inner",
        validate="one_to_one",
    )
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# bank payload assembly
# ---------------------------------------------------------------------------


def _make_shock_bank_payload(seed: int, n_child_shocks: int) -> Tuple[Dict[str, Any], str]:
    bank = ConvergenceShockBank.create(1, int(n_child_shocks), seed=int(seed), device="cpu", dtype=torch.float32)
    payload = {
        "seed": int(seed),
        "eps_x": bank.eps_x.detach().cpu(),
        "eps_z": bank.eps_z.detach().cpu(),
        "u_eta": bank.u_eta.detach().cpu(),
        "u_i": bank.u_i.detach().cpu(),
    }
    return payload, shock_bank_sha256(bank)


def _eta_share(states: torch.Tensor) -> Dict[str, float]:
    eta = states[:, FirmStateIndex().ETA].to(torch.float32)
    n = max(int(eta.numel()), 1)
    share1 = float((eta >= 0.5).to(torch.float32).sum().item()) / float(n)
    return {"eta0": 1.0 - share1, "eta1": share1}


def _source_share(source_ids: Sequence[int]) -> Dict[str, float]:
    total = max(len(source_ids), 1)
    counts = {SOURCE_ID_SIMULATED: 0, SOURCE_ID_COVERAGE: 0, SOURCE_ID_UNKNOWN: 0}
    for value in source_ids:
        key = int(value)
        counts[key if key in counts else SOURCE_ID_UNKNOWN] += 1
    return {
        "simulated": counts[SOURCE_ID_SIMULATED] / float(total),
        "coverage": counts[SOURCE_ID_COVERAGE] / float(total),
        "unknown": counts[SOURCE_ID_UNKNOWN] / float(total),
    }


def _infer_source_ids(frame: pd.DataFrame, count: int) -> List[int]:
    if "source_id" in frame.columns:
        return [int(value) for value in pd.to_numeric(frame["source_id"], errors="coerce").fillna(-1).tolist()]
    if "source" in frame.columns:
        mapping = {"simulated": SOURCE_ID_SIMULATED, "coverage": SOURCE_ID_COVERAGE}
        return [int(mapping.get(str(value).lower(), SOURCE_ID_UNKNOWN)) for value in frame["source"].tolist()]
    # Simulate-style panels are reference-policy simulation output by construction.
    return [SOURCE_ID_SIMULATED] * int(count)


def _assemble_payload(
    *,
    bank_id: str,
    bank_kind: str,
    tags: Sequence[str],
    seed: int,
    config_hash: Optional[str],
    parent_states: Optional[torch.Tensor],
    hatc_cal: Optional[torch.Tensor],
    lnk_cal: Optional[torch.Tensor],
    shock_payload: Dict[str, Any],
    shock_hash: str,
    source_id: Sequence[int],
    lineage_id: Sequence[str],
    pairing_keys: Dict[str, Any],
    row_weights: Optional[torch.Tensor],
    macro_anchor: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    n_rows = int(parent_states.shape[0]) if parent_states is not None else int(len(lineage_id))
    if row_weights is None:
        row_weights = torch.full((n_rows,), 1.0 / float(max(n_rows, 1)), dtype=torch.float32)
    payload: Dict[str, Any] = {
        "schema_version": BANK_SCHEMA_VERSION,
        "bank_id": bank_id,
        "bank_kind": bank_kind,
        "tags": list(tags),
        "seed": int(seed),
        "config_snapshot_hash": config_hash,
        "evaluation_code_commit": evaluation_code_commit(),
        "columns": list(STATE_COLUMNS),
        "n_rows": n_rows,
        "parent_states": parent_states,
        "hatc_cal": hatc_cal,
        "lnk_cal": lnk_cal,
        "shock_bank": shock_payload,
        "shock_bank_sha256": shock_hash,
        "row_weights": row_weights,
        "source_id": [int(value) for value in source_id],
        "lineage_id": [str(value) for value in lineage_id],
        "pairing_keys": pairing_keys,
        "macro_anchor": macro_anchor,
        "eta_share": _eta_share(parent_states) if parent_states is not None else {"eta0": None, "eta1": None},
        "source_share": _source_share(source_id),
        "metadata": metadata,
    }
    if extra:
        payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# Bank builders
# ---------------------------------------------------------------------------


def build_bank_a(
    firm_frame: pd.DataFrame,
    macro_frame: Optional[pd.DataFrame] = None,
    *,
    seed: int,
    n_rows: Optional[int] = None,
    config_hash: Optional[str] = None,
    n_child_shocks: int = 2,
    source_kind: str = TAG_REFERENCE_POLICY_DISTRIBUTION,
    checkpoint_ref: Optional[str] = None,
    exclude_lineages: Optional[Iterable[str]] = None,
    holdout_fraction: Optional[float] = None,
    holdout_salt: str = "evaluation_bank_v1/train-split",
) -> Dict[str, Any]:
    """Bank A — fixed natural firm parent states (natural eta distribution).

    Parent rows are extracted with an explicit branch filter
    (``select_parent_view_rows``), the realized macro context is joined by key
    (``Hatc``/``LnK`` -> ``hatc_cal``/``lnk_cal``; belief columns ``Hatcf``/
    ``LnKF`` stay in the firm state and are never mixed), sampling uses its own
    independent seed and eta is *not* rebalanced.  Training-data leakage is
    controlled by excluding lineages (explicit list and/or deterministic
    path/lineage hash split); the exclusion list is recorded for the manifest.
    """
    if {"b", "z", "ETA", "i", "x", "Hatcf", "LnKF"}.issubset(firm_frame.columns) is False:
        missing = sorted({"b", "z", "ETA", "i", "x", "Hatcf", "LnKF"} - set(firm_frame.columns))
        raise EvalBankError("MISSING_STATE_COLUMNS", f"firm frame is missing state columns: {missing}")

    parents = select_parent_view_rows(firm_frame)
    if macro_frame is not None:
        parents = attach_macro_context(parents, macro_frame)
        hatc_cal_values = parents["hatc_cal"].to_numpy(dtype=np.float32)
        lnk_cal_values = parents["lnk_cal"].to_numpy(dtype=np.float32)
    elif {"hatc_cal", "lnk_cal"}.issubset(firm_frame.columns):
        hatc_cal_values = parents["hatc_cal"].to_numpy(dtype=np.float32)
        lnk_cal_values = parents["lnk_cal"].to_numpy(dtype=np.float32)
    else:
        raise EvalBankError(
            REASON_MACRO_CONTEXT_MISSING,
            "Bank A requires a macro frame (realized Hatc/LnK) or hatc_cal/lnk_cal columns",
        )

    pairing = parents[list(PAIRING_KEY_COLUMNS)].copy()
    pairing["path"] = pd.to_numeric(pairing["path"]).astype(np.int64)
    pairing["t"] = pd.to_numeric(pairing["t"]).astype(np.int64)
    pairing["branch"] = pd.to_numeric(pairing["branch"]).astype(np.int64)
    pairing["economic_node_id"] = pd.to_numeric(pairing["economic_node_id"]).astype(np.int64)
    pairing["ID"] = pairing["ID"].astype(str)

    lineage_ids = [
        lineage_id_for(path, firm_id)
        for path, firm_id in zip(pairing["path"].tolist(), pairing["ID"].tolist())
    ]

    excluded: List[str] = []
    if exclude_lineages:
        excluded.extend(sorted({str(item) for item in exclude_lineages}))
    split_info = None
    if holdout_fraction is not None:
        split_info = hash_split_lineages(
            sorted(set(lineage_ids)), holdout_fraction=holdout_fraction, salt=holdout_salt
        )
        excluded.extend(split_info["excluded"])
    excluded_set = set(excluded)
    keep_mask = [lineage not in excluded_set for lineage in lineage_ids]
    excluded_present = sorted({lineage for lineage, keep in zip(lineage_ids, keep_mask) if not keep})

    indices = [index for index, keep in enumerate(keep_mask) if keep]
    if n_rows is not None and int(n_rows) < len(indices):
        rng = np.random.default_rng(int(seed))
        chosen = rng.choice(np.asarray(indices), size=int(n_rows), replace=False)
        indices = sorted(int(value) for value in chosen)

    states_frame = parents.iloc[indices][list(STATE_COLUMNS)].astype(np.float32)
    parent_states = torch.from_numpy(states_frame.to_numpy(dtype=np.float32).copy())
    hatc_cal = torch.from_numpy(np.asarray([hatc_cal_values[i] for i in indices], dtype=np.float32))
    lnk_cal = torch.from_numpy(np.asarray([lnk_cal_values[i] for i in indices], dtype=np.float32))

    source_ids = _infer_source_ids(parents, len(parents))
    source_ids = [source_ids[i] for i in indices]
    lineage_selected = [lineage_ids[i] for i in indices]
    pairing_keys = {
        "path": [int(pairing["path"].iloc[i]) for i in indices],
        "ID": [str(pairing["ID"].iloc[i]) for i in indices],
        "t": [int(pairing["t"].iloc[i]) for i in indices],
        "branch": [int(pairing["branch"].iloc[i]) for i in indices],
        "economic_node_id": [int(pairing["economic_node_id"].iloc[i]) for i in indices],
    }
    shock_payload, shock_hash = _make_shock_bank_payload(seed, n_child_shocks)

    metadata = {
        "sampling": {
            "seed": int(seed),
            "n_candidates": int(len(parents)),
            "n_selected": int(len(indices)),
            "independent_seed": True,
            "eta_rebalanced": False,
        },
        "branch_filter": "branch==-1 preferred, branch==0 fallback",
        "pairing_key_columns": list(PAIRING_KEY_COLUMNS),
        "macro_context": "macro Hatc/LnK joined by key into hatc_cal/lnk_cal",
        "train_lineage_exclusion": {
            "excluded": excluded_present,
            "requested": sorted(excluded_set),
            "hash_split": split_info,
        },
        "source_kind": source_kind,
        "checkpoint_ref": checkpoint_ref,
    }
    return _assemble_payload(
        bank_id="A",
        bank_kind="fixed-natural-firm-parents",
        tags=[TAG_REFERENCE_POLICY_DISTRIBUTION] if source_kind == TAG_REFERENCE_POLICY_DISTRIBUTION else [str(source_kind)],
        seed=seed,
        config_hash=config_hash,
        parent_states=parent_states,
        hatc_cal=hatc_cal,
        lnk_cal=lnk_cal,
        shock_payload=shock_payload,
        shock_hash=shock_hash,
        source_id=source_ids,
        lineage_id=lineage_selected,
        pairing_keys=pairing_keys,
        row_weights=None,
        macro_anchor=None,
        metadata=metadata,
    )


def build_bank_b(
    bank_a: Mapping[str, Any],
    *,
    seed: int,
    config_hash: Optional[str] = None,
    n_child_shocks: int = 2,
) -> Dict[str, Any]:
    """Bank B — balanced eta counterfactual views of the Bank A parent states.

    The same parent states are emitted twice with the ETA column covered by
    eta=0 and eta=1 (``counterfactual-eta-view``).  Both views are equally
    weighted; child rebuilding uses the parent-view eta financing gate
    ``b_next = eta_t * bp + (1 - eta_t) * b_t`` (see
    :func:`build_transition_for_checkpoint`).
    """
    base_states = bank_a["parent_states"]
    if base_states is None or base_states.ndim != 2:
        raise EvalBankError("INVALID_BANK_A", "build_bank_b requires Bank A parent_states [N,7]")
    n_base = int(base_states.shape[0])
    eta_index = FirmStateIndex().ETA

    zero_view = base_states.clone()
    zero_view[:, eta_index] = 0.0
    one_view = base_states.clone()
    one_view[:, eta_index] = 1.0
    parent_states = torch.cat([zero_view, one_view], dim=0)

    hatc_cal = torch.cat([bank_a["hatc_cal"], bank_a["hatc_cal"]], dim=0)
    lnk_cal = torch.cat([bank_a["lnk_cal"], bank_a["lnk_cal"]], dim=0)
    shock_payload, shock_hash = _make_shock_bank_payload(seed, n_child_shocks)

    pairing_keys = {
        key: list(values) + list(values) for key, values in bank_a["pairing_keys"].items()
    }
    pairing_keys["eta_view"] = ["eta0"] * n_base + ["eta1"] * n_base
    pairing_keys["source_row_index"] = list(range(n_base)) + list(range(n_base))

    metadata = {
        "parent_bank": "A",
        "counterfactual-eta-view": True,
        "equal_view_weights": True,
        "financing_gate": "b_next = eta_t * bp + (1 - eta_t) * b_t (parent eta_t per view)",
        "source_kind": bank_a["metadata"].get("source_kind"),
        "checkpoint_ref": bank_a["metadata"].get("checkpoint_ref"),
        "train_lineage_exclusion": bank_a["metadata"].get("train_lineage_exclusion"),
    }
    return _assemble_payload(
        bank_id="B",
        bank_kind="balanced-eta-counterfactual-views",
        tags=[TAG_COUNTERFACTUAL_ETA_VIEW],
        seed=seed,
        config_hash=config_hash if config_hash is not None else bank_a.get("config_snapshot_hash"),
        parent_states=parent_states,
        hatc_cal=hatc_cal,
        lnk_cal=lnk_cal,
        shock_payload=shock_payload,
        shock_hash=shock_hash,
        source_id=list(bank_a["source_id"]) * 2,
        lineage_id=list(bank_a["lineage_id"]) * 2,
        pairing_keys=pairing_keys,
        row_weights=torch.full((2 * n_base,), 1.0 / float(2 * n_base), dtype=torch.float32),
        macro_anchor=None,
        metadata=metadata,
        extra={"eta_share": {"eta0": 0.5, "eta1": 0.5}},
    )


def derive_macro_anchors(
    bank_d: Mapping[str, Any],
    *,
    labels: Sequence[str] = ("Recession", "Normal", "Expansion"),
    quantiles: Sequence[float] = (0.10, 0.50, 0.90),
) -> Dict[str, Dict[str, float]]:
    """Macro anchors as *joint* Bank D samples at conditional x quantiles.

    For each label the row whose ``x_t`` sits at the requested quantile of the
    Bank D x distribution is taken **as a whole** (x, hatcf, lnkf, hatc, lnk),
    so anchors are macro combinations the economy actually visits.  Independently
    splicing per-column quantiles would fabricate out-of-sample macro
    combinations and is explicitly avoided.
    """
    if len(labels) != len(quantiles):
        raise EvalBankError("INVALID_ANCHOR_SPEC", "labels and quantiles must have equal length")
    x_values = np.asarray(bank_d.get("x_t") or [], dtype=np.float64)
    if x_values.size == 0:
        raise EvalBankError("EMPTY_BANK_D", "derive_macro_anchors requires non-empty Bank D rows")
    order = np.argsort(x_values, kind="stable")
    n = int(x_values.size)
    anchors: Dict[str, Dict[str, float]] = {}
    for label, quantile in zip(labels, quantiles):
        position = int(round(float(quantile) * (n - 1)))
        index = int(order[min(max(position, 0), n - 1)])
        anchors[str(label)] = {
            "label": str(label),
            "quantile": float(quantile),
            "x": float(bank_d["x_t"][index]),
            "hatcf": float(bank_d["hatcf_t"][index]),
            "lnkf": float(bank_d["lnkf_t"][index]),
            "hatc_cal": float(bank_d["hatc_t"][index]),
            "lnk_cal": float(bank_d["lnk_t"][index]),
            "source_row": index,
            "derivation": "joint Bank D row at conditional x quantile",
        }
    return anchors


def build_bank_c(
    *,
    seed: int,
    macro_anchors: Mapping[str, Mapping[str, float]],
    b_values: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
    z_values: Sequence[float] = (-2.5, -1.5, 0.0, 1.5, 2.5),
    i_values: Sequence[float] = (0.10, 0.25, 0.45),
    eta_values: Sequence[float] = (0.0, 1.0),
    config_hash: Optional[str] = None,
    n_child_shocks: int = 2,
) -> Dict[str, Any]:
    """Bank C — fixed boundary/stress states (``out-of-distribution-stress``).

    Fixed synthetic stress states: b endpoints and high leverage, z tails inside
    ``[-2.5, 2.5]``, ``i in {0.10, 0.25, 0.45}``, ``eta in {0, 1}`` and the three
    macro anchors Recession/Normal/Expansion (taken from Bank D joint samples via
    :func:`derive_macro_anchors`, never spliced arbitrarily).
    """
    anchor_items = list(macro_anchors.items())
    if not anchor_items:
        raise EvalBankError("INVALID_ANCHOR_SPEC", "build_bank_c requires at least one macro anchor")

    rows: List[List[float]] = []
    hatc_values: List[float] = []
    lnk_values: List[float] = []
    anchor_labels: List[str] = []
    for label, anchor in anchor_items:
        for b_value in b_values:
            for z_value in z_values:
                for i_value in i_values:
                    for eta_value in eta_values:
                        rows.append(
                            [
                                float(b_value),
                                float(z_value),
                                float(eta_value),
                                float(i_value),
                                float(anchor["x"]),
                                float(anchor["hatcf"]),
                                float(anchor["lnkf"]),
                            ]
                        )
                        hatc_values.append(float(anchor["hatc_cal"]))
                        lnk_values.append(float(anchor["lnk_cal"]))
                        anchor_labels.append(str(label))

    parent_states = torch.tensor(rows, dtype=torch.float32)
    hatc_cal = torch.tensor(hatc_values, dtype=torch.float32)
    lnk_cal = torch.tensor(lnk_values, dtype=torch.float32)
    shock_payload, shock_hash = _make_shock_bank_payload(seed, n_child_shocks)
    n_rows = int(parent_states.shape[0])
    pairing_keys = {
        "path": [-1] * n_rows,
        "ID": [f"stress_{index}" for index in range(n_rows)],
        "t": [0] * n_rows,
        "branch": [-1] * n_rows,
        "economic_node_id": [-1] * n_rows,
        "anchor_label": anchor_labels,
    }
    metadata = {
        "stress_grid": {
            "b_values": [float(value) for value in b_values],
            "z_values": [float(value) for value in z_values],
            "i_values": [float(value) for value in i_values],
            "eta_values": [float(value) for value in eta_values],
        },
        "macro_anchor_source": "Bank D joint samples at conditional x quantiles",
        "grid_is_synthetic": True,
    }
    return _assemble_payload(
        bank_id="C",
        bank_kind="boundary-stress-grid",
        tags=[TAG_OUT_OF_DISTRIBUTION_STRESS],
        seed=seed,
        config_hash=config_hash,
        parent_states=parent_states,
        hatc_cal=hatc_cal,
        lnk_cal=lnk_cal,
        shock_payload=shock_payload,
        shock_hash=shock_hash,
        source_id=[SOURCE_ID_UNKNOWN] * n_rows,
        lineage_id=[f"synthetic::stress_{index}" for index in range(n_rows)],
        pairing_keys=pairing_keys,
        row_weights=None,
        macro_anchor={str(label): dict(anchor) for label, anchor in anchor_items},
        metadata=metadata,
    )


def build_bank_d(
    macro_frame: pd.DataFrame,
    *,
    seed: int,
    n_rows: Optional[int] = None,
    config_hash: Optional[str] = None,
    n_child_shocks: int = 2,
    exclude_paths: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Bank D — macro holdout parent-child rows (paired by key only).

    Rows carry ``{x_t, x_t1, Hatcf_t, LnKF_t, Hatc_t, LnK_t, Hatc_t1, LnK_t1}``
    with pairing keys ``(path, t, branch, economic_node_id)``.  Pairing is a key
    join on ``(path, t + 1, branch)`` — never row adjacency.
    """
    pairs = pair_macro_parent_child(macro_frame)
    if exclude_paths:
        excluded = {int(value) for value in exclude_paths}
        pairs = pairs.loc[~pairs["path"].isin(excluded)].reset_index(drop=True)
    if pairs.empty:
        raise EvalBankError("EMPTY_BANK_D", "no macro parent-child pairs available for Bank D")

    indices = list(range(int(len(pairs))))
    if n_rows is not None and int(n_rows) < len(indices):
        rng = np.random.default_rng(int(seed))
        chosen = rng.choice(np.asarray(indices), size=int(n_rows), replace=False)
        indices = sorted(int(value) for value in chosen)

    selected = pairs.iloc[indices].reset_index(drop=True)
    shock_payload, shock_hash = _make_shock_bank_payload(seed, n_child_shocks)
    pairing_keys = {
        "path": [int(value) for value in selected["path"].tolist()],
        "t": [int(value) for value in selected["t"].tolist()],
        "branch": [int(value) for value in selected["branch"].tolist()],
        "economic_node_id": [int(value) for value in selected["economic_node_id"].tolist()],
        "t1": [int(value) for value in selected["t1"].tolist()],
        "economic_node_id_child": [int(value) for value in selected["economic_node_id_child"].tolist()],
    }
    n_rows_out = int(len(selected))
    payload = _assemble_payload(
        bank_id="D",
        bank_kind="macro-holdout-pairs",
        tags=["macro-holdout"],
        seed=seed,
        config_hash=config_hash,
        parent_states=None,
        hatc_cal=None,
        lnk_cal=None,
        shock_payload=shock_payload,
        shock_hash=shock_hash,
        source_id=[SOURCE_ID_UNKNOWN] * n_rows_out,
        lineage_id=[f"path::{int(path)}" for path in selected["path"].tolist()],
        pairing_keys=pairing_keys,
        row_weights=None,
        macro_anchor=None,
        metadata={
            "pairing": "key join on (path, t+1, branch); row adjacency forbidden",
            "excluded_paths": sorted({int(value) for value in (exclude_paths or [])}),
            "sampling": {"seed": int(seed), "n_selected": n_rows_out},
        },
        extra={
            "x_t": [float(value) for value in selected["x_t"].tolist()],
            "x_t1": [float(value) for value in selected["x_t1"].tolist()],
            "hatcf_t": [float(value) for value in selected["hatcf_t"].tolist()],
            "lnkf_t": [float(value) for value in selected["lnkf_t"].tolist()],
            "hatc_t": [float(value) for value in selected["hatc_t"].tolist()],
            "lnk_t": [float(value) for value in selected["lnk_t"].tolist()],
            "hatc_t1": [float(value) for value in selected["hatc_t1"].tolist()],
            "lnk_t1": [float(value) for value in selected["lnk_t1"].tolist()],
        },
    )
    payload["columns"] = [
        "x_t",
        "x_t1",
        "hatcf_t",
        "lnkf_t",
        "hatc_t",
        "lnk_t",
        "hatc_t1",
        "lnk_t1",
    ]
    return payload


def build_bank_e(
    *,
    seed: int,
    n_firms: int = 64,
    n_paths: int = 4,
    horizon: int = 20,
    economic_config: Optional[AnalysisEconomicConfig] = None,
    config_hash: Optional[str] = None,
    n_child_shocks: int = 2,
    leverage_variants: Sequence[str] = DEFAULT_LEVERAGE_VARIANTS,
    leverage_factors: Sequence[float] = (0.5, 1.0, 1.5),
    k0_total: float = 1.0,
) -> Dict[str, Any]:
    """Bank E — fixed initial cross-section + fixed keyed shock specification.

    The initial cross-section ``{b0, z0, eta0, i0, K0, x0}`` is drawn from the
    bank seed; the three initial-leverage variants (Low/Baseline/High) scale
    ``b0`` only, so the capital totals stay comparable across variants.  Rollout
    shocks follow the ``stable_firm_identity`` keyed-RNG convention of
    ``data/simulate_ts_parallel`` (key = (seed, path, economic_time, branch,
    identity, shock_type)) and are drawn by the rollout driver, not precomputed
    here.  The initial h-star is *not* solved here: every checkpoint calibrates
    it under the same rule and records its own difference.
    """
    if len(leverage_variants) != len(leverage_factors):
        raise EvalBankError("INVALID_LEVERAGE_VARIANTS", "leverage_variants and leverage_factors must align")
    config = economic_config
    rng = np.random.default_rng(int(seed))

    firms_per_path = int(n_firms) // max(int(n_paths), 1)
    total_firms = firms_per_path * max(int(n_paths), 1)
    path_ids = np.repeat(np.arange(int(n_paths), dtype=np.int64), firms_per_path)

    b0 = rng.uniform(0.0, 1.0, size=total_firms).astype(np.float32)
    z0 = rng.normal(0.0, 1.0, size=total_firms).astype(np.float32)
    if config is not None:
        z0 = (float(config.Z0) + float(config.ALPHA_Z) * z0).astype(np.float32)
    eta_threshold = float(config.ZETA) if config is not None else 0.5
    i_threshold = float(config.I_THRESHOLD) if config is not None else 0.5
    eta0 = (rng.uniform(0.0, 1.0, size=total_firms) < eta_threshold).astype(np.float32)
    i0 = (rng.uniform(0.0, 1.0, size=total_firms) * i_threshold).astype(np.float32)
    weights = rng.uniform(0.5, 1.5, size=total_firms).astype(np.float64)
    k0 = (weights / weights.sum() * float(k0_total)).astype(np.float32)
    x0 = np.full(total_firms, float(config.XBAR) if config is not None else 0.0, dtype=np.float32)

    variants_payload: Dict[str, Any] = {}
    for name, factor in zip(leverage_variants, leverage_factors):
        variants_payload[str(name)] = {
            "leverage_factor": float(factor),
            "construction": "b0_variant = factor * b0 (K0 unchanged -> capital totals comparable)",
            "b0": torch.from_numpy((b0 * float(factor)).astype(np.float32).copy()),
            "z0": torch.from_numpy(z0.copy()),
            "eta0": torch.from_numpy(eta0.copy()),
            "i0": torch.from_numpy(i0.copy()),
            "K0": torch.from_numpy(k0.copy()),
            "x0": torch.from_numpy(x0.copy()),
            "K0_total": float(k0.sum()),
        }

    shock_spec = {
        "rng_mode": "stable_firm_identity",
        "seed": int(seed),
        "key_fields": ["seed", "path", "economic_time", "branch", "identity", "shock_type"],
        "shock_types": {"eps_x": 1, "eps_z": 2, "u_eta": 3, "u_i": 4},
        "identity": {"macro": 0, "firm": "firm_id"},
        "source": "data/simulate_ts_parallel._keyed_uniform/_keyed_normal",
    }
    shock_payload, shock_hash = _make_shock_bank_payload(seed, n_child_shocks)
    n_rows = int(total_firms)
    pairing_keys = {
        "path": [int(value) for value in path_ids.tolist()],
        "ID": [str(index) for index in range(n_rows)],
        "t": [0] * n_rows,
        "branch": [-1] * n_rows,
        "economic_node_id": [0] * n_rows,
    }
    payload = _assemble_payload(
        bank_id="E",
        bank_kind="fixed-rollout",
        tags=["fixed-rollout", "initial-leverage-variants"],
        seed=seed,
        config_hash=config_hash,
        parent_states=None,
        hatc_cal=None,
        lnk_cal=None,
        shock_payload=shock_payload,
        shock_hash=shock_hash,
        source_id=[SOURCE_ID_UNKNOWN] * n_rows,
        lineage_id=[lineage_id_for(path, index) for path, index in zip(path_ids.tolist(), range(n_rows))],
        pairing_keys=pairing_keys,
        row_weights=None,
        macro_anchor=None,
        metadata={
            "n_firms": n_rows,
            "n_paths": int(n_paths),
            "firms_per_path": int(firms_per_path),
            "horizon": int(horizon),
            "shock_spec": shock_spec,
            "leverage_variants": {str(name): float(factor) for name, factor in zip(leverage_variants, leverage_factors)},
            "capital_totals_comparable": True,
            "initial_hstar": "not solved here; each checkpoint calibrates under the same rule and records its difference",
        },
        extra={"variants": variants_payload, "shock_spec": shock_spec, "horizon": int(horizon)},
    )
    payload["columns"] = ["b0", "z0", "eta0", "i0", "K0", "x0"]
    return payload


# ---------------------------------------------------------------------------
# per-checkpoint transition rebuilding
# ---------------------------------------------------------------------------


@dataclass
class CheckpointTransition:
    """Children rebuilt by one checkpoint's FC1/SDF under the bank's fixed shocks."""

    parent_states: torch.Tensor
    children: torch.Tensor
    m_raw: torch.Tensor
    m_used: torch.Tensor
    branch_weights: torch.Tensor
    metadata: Dict[str, Any]

    def stacked_children(self) -> torch.Tensor:
        return self.children

    def stacked_m_raw(self) -> torch.Tensor:
        return self.m_raw

    def stacked_m_used(self) -> torch.Tensor:
        return self.m_used

    def child_b_with_policy(self, bp: torch.Tensor) -> torch.Tensor:
        """Realized child leverage under the parent-view financing gate.

        ``b_next = eta_t * bp + (1 - eta_t) * b_t`` where ``eta_t`` is the
        *parent* eta (the counterfactual view's ETA for Bank B).  Returns shape
        ``[N, 2J, 1]``.
        """
        bp_col = torch.as_tensor(bp, dtype=self.parent_states.dtype, device=self.parent_states.device)
        bp_col = bp_col.reshape(self.parent_states.shape[0], 1)
        b_next = apply_refinancing_policy(
            self.parent_states[:, :1],
            bp_col,
            self.parent_states[:, FirmStateIndex().ETA:FirmStateIndex().ETA + 1],
        )
        return b_next.unsqueeze(1).expand(-1, int(self.children.shape[1]), -1).contiguous()


def build_transition_for_checkpoint(
    bank_entry: Mapping[str, Any],
    sdf_fc1_model: torch.nn.Module,
    hyperparams: HyperParams,
    economic_config: AnalysisEconomicConfig,
    *,
    eta_override: Optional[float] = None,
    return_physical: bool = True,
) -> CheckpointTransition:
    """Rebuild children (with ``hatcf_next``/``lnkf_next`` and SDF ``M``) for one
    checkpoint under the bank's fixed shocks and macro context.

    Reuses the existing mechanisms of ``analysis/convergence_transition``
    (``ConvergenceShockBank`` + ``build_child_exogenous_bundle``) and the exact
    eta expansion of ``utils.firm_transition``.  The child leverage gate uses the
    parent-view eta (``eta_override`` replaces the bank's ETA column, e.g. to
    force one counterfactual view).
    """
    states = torch.as_tensor(bank_entry["parent_states"]).to(torch.float32)
    if states.ndim != 2 or states.shape[1] != len(STATE_COLUMNS):
        raise EvalBankError("INVALID_BANK_ENTRY", "bank_entry parent_states must have shape [N,7]")
    if eta_override is not None:
        states = states.clone()
        states[:, FirmStateIndex().ETA] = float(eta_override)

    shock_payload = bank_entry["shock_bank"]
    full_bank = ConvergenceShockBank(
        eps_x=torch.as_tensor(shock_payload["eps_x"]),
        eps_z=torch.as_tensor(shock_payload["eps_z"]),
        u_eta=torch.as_tensor(shock_payload["u_eta"]),
        u_i=torch.as_tensor(shock_payload["u_i"]),
        seed=int(shock_payload["seed"]),
    )
    n_parent = int(states.shape[0])
    reference_index = torch.zeros(n_parent, dtype=torch.long)
    shock_bank = full_bank.gather(reference_index)

    hatc_cal = torch.as_tensor(bank_entry["hatc_cal"]).to(torch.float32).reshape(-1)
    lnk_cal = torch.as_tensor(bank_entry["lnk_cal"]).to(torch.float32).reshape(-1)
    macro = MacroTransitionContext(hatc_cal=hatc_cal, lnk_cal=lnk_cal)

    bundle = build_child_exogenous_bundle(
        sdf_fc1_model,
        states,
        macro,
        shock_bank,
        economic_config=economic_config,
        return_physical=return_physical,
    )
    n_child = int(bundle.z_next.shape[1])
    continuous_children = torch.stack(
        [
            states[:, 0].reshape(n_parent, 1).expand(n_parent, n_child),
            bundle.z_next[..., 0],
            torch.zeros_like(bundle.eta_next[..., 0]),
            bundle.i_next[..., 0],
            bundle.x_next[..., 0],
            bundle.hatcf_next[..., 0],
            bundle.lnkf_next[..., 0],
        ],
        dim=-1,
    )
    use_clipped_m = bool(getattr(hyperparams, "pv_use_clipped_m", True))
    m_lo = float(getattr(hyperparams, "pv_m_clamp_min", 0.7))
    m_hi = float(getattr(hyperparams, "pv_m_clamp_max", 1.3))
    m_raw = bundle.m_raw
    m_used = m_raw.clamp(m_lo, m_hi) if use_clipped_m else m_raw

    expansion = expand_children_exact_eta_tensor(
        continuous_children,
        zeta=float(economic_config.ZETA),
        child_weights=bundle.branch_weights,
    )
    children = expansion.children_tensor
    if children is None:
        raise RuntimeError("exact eta expansion did not return children_tensor")
    m_raw_expanded = (
        m_raw.unsqueeze(2).expand(-1, -1, 2, -1).reshape(n_parent, 2 * n_child, 1)
    )
    m_used_expanded = (
        m_used.unsqueeze(2).expand(-1, -1, 2, -1).reshape(n_parent, 2 * n_child, 1)
    )

    metadata = {
        "builder": "bank shock bank + build_child_exogenous_bundle + exact eta expansion",
        "bank_id": bank_entry.get("bank_id"),
        "shock_bank_sha256": shock_bank_sha256(full_bank),
        "bank_shock_bank_sha256": bank_entry.get("shock_bank_sha256"),
        "n_child_shocks": n_child,
        "continuous_child_count": n_child,
        "expanded_child_count": 2 * n_child,
        "eta_integration_mode": "exact",
        "eta_probability": float(economic_config.ZETA),
        "eta_override": eta_override,
        "parent_eta_share": _eta_share(states),
        "macro_context": "bank per-parent hatc_cal/lnk_cal (realized aggregate)",
        "m_source": "sdf_fc1.forward_step",
        "m_mode": "clipped_train_m" if use_clipped_m else "raw_sdf_m",
        "financing_gate": "b_next = eta_t * bp + (1 - eta_t) * b_t (parent eta_t)",
    }
    return CheckpointTransition(
        parent_states=states,
        children=children,
        m_raw=m_raw_expanded,
        m_used=m_used_expanded,
        branch_weights=expansion.branch_weights,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def _bank_relative_path(bank_key: str) -> str:
    return f"bank_{str(bank_key).lower()}/parents.pt"


def save_bank(
    root: str | Path,
    banks: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """Write ``bank_manifest.json`` + one ``.pt`` per bank under ``root``.

    An existing non-empty ``root`` is never overwritten silently: without an
    explicit ``version`` upgrade this raises ``BANK_ROOT_EXISTS``.  With an
    explicit ``version`` that differs from the previously recorded bank version
    the write proceeds and the returned payload carries an explicit
    *incomparable* warning (results across bank versions must not be compared).
    """
    root_path = Path(root)
    requested_version = str(version or manifest.get("bank_version") or DEFAULT_BANK_VERSION)
    warnings: List[str] = []
    incomparable = False

    if root_path.exists() and any(root_path.iterdir()):
        previous_version = None
        previous_manifest_path = root_path / BANK_MANIFEST_FILENAME
        if previous_manifest_path.is_file():
            try:
                previous_version = json.loads(previous_manifest_path.read_text(encoding="utf-8")).get("bank_version")
            except Exception:
                previous_version = None
        if version is None:
            raise EvalBankError(
                REASON_BANK_ROOT_EXISTS,
                f"bank root already exists and is not empty: {root_path}; "
                "pass an explicit version= to upgrade the bank version",
            )
        if previous_version is not None and str(requested_version) == str(previous_version):
            raise EvalBankError(
                REASON_BANK_VERSION_NOT_UPGRADED,
                f"bank root {root_path} already holds bank_version={previous_version!r}; "
                "an upgrade must declare a new version (e.g. evaluation_bank_v2)",
            )
        incomparable = True
        warnings.append(
            f"bank root {root_path} already existed (previous bank_version={previous_version!r}); "
            f"written as bank_version={requested_version!r}. Results from different bank "
            "versions are NOT comparable."
        )
    root_path.mkdir(parents=True, exist_ok=True)

    manifest_banks_extra = dict(manifest.get("banks") or {})
    bank_summaries: Dict[str, Any] = {}
    for key in sorted(banks.keys()):
        payload = dict(banks[key])
        bank_key = str(key).upper()
        relative = _bank_relative_path(bank_key)
        target = root_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)
        extra_summary = dict(manifest_banks_extra.get(bank_key) or manifest_banks_extra.get(key) or {})
        bank_summaries[bank_key] = {
            **extra_summary,
            "file": relative,
            "sha256": sha256_file(target),
            "content_sha256": content_sha256(payload),
            "schema_version": payload.get("schema_version", BANK_SCHEMA_VERSION),
            "bank_kind": payload.get("bank_kind"),
            "tags": list(payload.get("tags") or []),
            "seed": payload.get("seed"),
            "n_rows": payload.get("n_rows"),
            "eta_share": payload.get("eta_share"),
            "source_share": payload.get("source_share"),
            "columns": payload.get("columns"),
        }

    seeds = dict(manifest.get("seeds") or {})
    for bank_key, summary in bank_summaries.items():
        seeds.setdefault(f"bank_{bank_key.lower()}", summary.get("seed"))
    source = dict(manifest.get("source") or {})
    source.setdefault("kind", None)
    source.setdefault("checkpoint_ref", None)
    source.setdefault("train_lineage_exclusion", [])

    manifest_out: Dict[str, Any] = {
        "bank_version": requested_version,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evaluation_code_commit": manifest.get("evaluation_code_commit") or evaluation_code_commit(),
        "seeds": seeds,
        "config_snapshot_hash": manifest.get("config_snapshot_hash"),
        "source": source,
        "banks": bank_summaries,
        "incomparable_with_previous_version": bool(incomparable),
        "warnings": warnings,
        "schema_version": MANIFEST_SCHEMA_VERSION,
    }
    for key, value in manifest.items():
        if key not in manifest_out:
            manifest_out[key] = value

    manifest_path = root_path / BANK_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest_out, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "root": str(root_path),
        "manifest": manifest_out,
        "warnings": warnings,
        "incomparable": bool(incomparable),
    }


def load_bank(root: str | Path, *, verify: bool = True) -> Dict[str, Any]:
    """Load a saved bank root and re-verify every bank file sha256."""
    root_path = Path(root)
    manifest_path = root_path / BANK_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise EvalBankError(
            REASON_MISSING_BANK_MANIFEST,
            f"missing {BANK_MANIFEST_FILENAME} under bank root: {root_path}",
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    banks: Dict[str, Any] = {}
    for bank_key, info in sorted((manifest.get("banks") or {}).items()):
        bank_path = root_path / str(info["file"])
        if verify:
            digest = sha256_file(bank_path)
            if digest != info.get("sha256"):
                raise EvalBankError(
                    REASON_BANK_FILE_HASH_MISMATCH,
                    f"bank file {bank_path} sha256 mismatch: recorded {info.get('sha256')}, computed {digest}",
                )
        banks[str(bank_key).upper()] = torch.load(bank_path, map_location="cpu", weights_only=False)
    return {"root": str(root_path), "manifest": manifest, "banks": banks}
