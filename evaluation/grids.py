from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class ReferenceFirmState:
    eta: float
    i_low: float
    i_mid: float
    i_high: float
    x: float
    hatcf: float
    lnkf: float
    hatc_cal: float
    lnk_cal: float
    n_parent_rows: int
    source: str
    macro_source: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenFirmGrid:
    b_values: np.ndarray
    z_values: np.ndarray
    mesh_b: np.ndarray
    mesh_z: np.ndarray
    base_states: torch.Tensor

    @property
    def shape(self) -> Tuple[int, int]:
        return self.mesh_b.shape


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "B": "b",
        "Z": "z",
        "eta": "ETA",
        "Eta": "ETA",
        "hatcf": "Hatcf",
        "lnkf": "LnKF",
        "Hatc": "hatc_cal",
        "HATC": "hatc_cal",
        "Hatc_cal": "hatc_cal",
        "LnK": "lnk_cal",
        "LNK": "lnk_cal",
        "LnK_cal": "lnk_cal",
    }
    rename = {old: new for old, new in aliases.items() if old in df.columns and new not in df.columns}
    return df.rename(columns=rename)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("--firm-data must be a .pkl, .pickle, or .csv file")


def _attach_calculated_macro(
    df: pd.DataFrame,
    firm_path: Path,
    macro_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str]:
    df = _normalize_columns(df)
    if {"hatc_cal", "lnk_cal"}.issubset(df.columns):
        return df, "embedded_in_firm_data"
    if macro_path is not None:
        candidates = [Path(macro_path).expanduser().resolve()]
        if not candidates[0].exists():
            raise FileNotFoundError(f"Explicit macro data not found: {candidates[0]}")
    else:
        candidates = [firm_path.with_name(f"{firm_path.stem}_macro{firm_path.suffix}")]
        if firm_path.stem.endswith("_firm"):
            prefix = firm_path.stem.removesuffix("_firm")
            candidates.append(firm_path.with_name(f"{prefix}_macro{firm_path.suffix}"))
        candidates = list(dict.fromkeys(candidates))
    existing = [candidate for candidate in candidates if candidate.exists()]
    if not existing:
        attempted = ", ".join(str(candidate) for candidate in candidates)
        raise ValueError(
            "Reference firm data is missing calculated macro state Hatc/LnK and "
            f"no macro file was found; attempted: {attempted}"
        )
    macro_path = existing[0]
    macro = _normalize_columns(_read_frame(macro_path))
    calculated = [name for name in ("hatc_cal", "lnk_cal") if name in macro.columns]
    if len(calculated) != 2:
        raise ValueError(f"Sibling macro data is missing Hatc/LnK: {macro_path}")
    key_options = [
        [name for name in ("path", "t", "branch") if name in df.columns and name in macro.columns],
        [name for name in ("path", "t") if name in df.columns and name in macro.columns],
    ]
    best = None
    best_complete = -1
    for keys in key_options:
        if not keys:
            continue
        candidate = df.merge(
            macro[keys + calculated].drop_duplicates(keys),
            on=keys,
            how="left",
        )
        complete = int(candidate[calculated].notna().all(axis=1).sum())
        if complete > best_complete:
            best = candidate
            best_complete = complete
        if complete == len(candidate):
            break
    if best is None or best_complete == 0:
        raise ValueError(
            "Could not align sibling macro Hatc/LnK with firm data using path/t/branch keys"
        )
    return best, str(macro_path)


def select_parent_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)
    if "branch" in df.columns:
        branch = pd.to_numeric(df["branch"], errors="coerce")
        if bool((branch < 0).any()):
            return df.loc[branch < 0].copy()
        if bool((branch == 0).any()):
            return df.loc[branch == 0].copy()
    if "t" in df.columns:
        text_t = df["t"].astype(str)
        if bool((text_t == "t").any()):
            return df.loc[text_t == "t"].copy()
    return df.copy()


def load_reference_state(
    path: str | Path,
    *,
    macro_path: str | Path | None = None,
) -> tuple[pd.DataFrame, ReferenceFirmState]:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reference firm dataframe not found: {path}")
    df = _read_frame(path)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Reference firm dataframe is empty or invalid")
    df, macro_source = _attach_calculated_macro(df, path, macro_path)
    parents = select_parent_rows(df)
    required = ["b", "z", "ETA", "i", "x", "Hatcf", "LnKF", "hatc_cal", "lnk_cal"]
    missing = [name for name in required if name not in parents.columns]
    if missing:
        raise ValueError(f"Reference parent rows are missing required columns: {missing}")
    numeric = parents[required].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
    numeric = numeric.loc[finite]
    if numeric.empty:
        raise ValueError("Reference dataframe has no finite parent states")
    i_quantiles = numeric["i"].quantile([0.10, 0.50, 0.90])
    reference = ReferenceFirmState(
        eta=1.0,
        i_low=float(i_quantiles.loc[0.10]),
        i_mid=float(i_quantiles.loc[0.50]),
        i_high=float(i_quantiles.loc[0.90]),
        x=float(numeric["x"].median()),
        hatcf=float(numeric["Hatcf"].median()),
        lnkf=float(numeric["LnKF"].median()),
        hatc_cal=float(numeric["hatc_cal"].median()),
        lnk_cal=float(numeric["lnk_cal"].median()),
        n_parent_rows=int(len(numeric)),
        source=str(path),
        macro_source=macro_source,
    )
    return df, reference


def build_frozen_grid(
    reference: ReferenceFirmState,
    *,
    b_min: float,
    b_max: float,
    b_points: int,
    z_min: float,
    z_max: float,
    z_points: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> FrozenFirmGrid:
    if b_points < 2 or z_points < 2:
        raise ValueError("b_points and z_points must both be at least 2")
    if not b_min < b_max or not z_min < z_max:
        raise ValueError("Grid lower bounds must be strictly below upper bounds")
    b_values = np.linspace(float(b_min), float(b_max), int(b_points), dtype=np.float64)
    z_values = np.linspace(float(z_min), float(z_max), int(z_points), dtype=np.float64)
    mesh_b, mesh_z = np.meshgrid(b_values, z_values, indexing="ij")
    states = torch.tensor(
        np.column_stack(
            [
                mesh_b.reshape(-1),
                mesh_z.reshape(-1),
                np.full(mesh_b.size, reference.eta),
                np.full(mesh_b.size, reference.i_mid),
                np.full(mesh_b.size, reference.x),
                np.full(mesh_b.size, reference.hatcf),
                np.full(mesh_b.size, reference.lnkf),
            ]
        ),
        dtype=dtype,
        device=device,
    )
    return FrozenFirmGrid(
        b_values=b_values,
        z_values=z_values,
        mesh_b=mesh_b,
        mesh_z=mesh_z,
        base_states=states,
    )
