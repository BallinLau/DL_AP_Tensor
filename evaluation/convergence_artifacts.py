from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EPISODE_DIR_RE = re.compile(r"^ep(?P<episode>\d+)_eta_matrix(?:_|$)")
STAGE_FIRM_RE = re.compile(r"^ep(?P<episode>\d+)_stage_(?P<stage>.+)\.pkl$")


@dataclass(frozen=True)
class EpisodeEvaluation:
    episode: int
    root: Path

    def eta_dir(self, eta: int) -> Path:
        return self.root / f"eta{int(eta)}"


@dataclass(frozen=True)
class SurfaceData:
    values: np.ndarray
    b: np.ndarray
    z: np.ndarray
    path: Path


def parse_episode_selection(spec: str | None) -> set[int] | None:
    if spec is None or not spec.strip():
        return None
    selected: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            start_text, stop_text = token.split(":", 1)
            start, stop = int(start_text), int(stop_text)
            step = 1 if stop >= start else -1
            selected.update(range(start, stop + step, step))
        else:
            selected.add(int(token))
    return selected


def discover_episode_dirs(
    run_root: str | Path,
    *,
    episodes: Iterable[int] | None = None,
) -> tuple[list[EpisodeEvaluation], list[str]]:
    """Discover evaluator matrix directories in numeric episode order.

    If multiple timestamped directories exist for one episode, choose the
    lexicographically last directory and report the ambiguity. Timestamped
    evaluator directories sort chronologically without relying on mtime.
    """
    root = Path(run_root).expanduser().resolve()
    requested = None if episodes is None else set(int(value) for value in episodes)
    candidates: dict[int, list[Path]] = {}
    if root.is_dir():
        for path in root.rglob("ep*_eta_matrix*"):
            if not path.is_dir():
                continue
            match = EPISODE_DIR_RE.match(path.name)
            if match is None or not (path / "metadata.json").is_file():
                continue
            episode = int(match.group("episode"))
            if requested is None or episode in requested:
                candidates.setdefault(episode, []).append(path.resolve())

    warnings: list[str] = []
    discovered: list[EpisodeEvaluation] = []
    for episode in sorted(candidates):
        paths = sorted(candidates[episode], key=lambda item: str(item))
        chosen = paths[-1]
        if len(paths) > 1:
            warnings.append(
                f"Episode {episode} has {len(paths)} evaluator directories; "
                f"selected lexicographically last: {chosen}"
            )
        discovered.append(EpisodeEvaluation(episode=episode, root=chosen))
    return discovered, warnings


def load_metadata(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata must be a JSON object: {path}")
    return value


def load_surface_csv(path: str | Path) -> SurfaceData:
    path = Path(path)
    frame = pd.read_csv(path, index_col=0)
    if frame.empty or frame.shape[1] == 0:
        raise ValueError(f"surface CSV is empty: {path}")
    try:
        b = pd.to_numeric(frame.index, errors="raise").to_numpy(dtype=np.float64)
        z = pd.to_numeric(frame.columns, errors="raise").to_numpy(dtype=np.float64)
        values = frame.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"surface CSV is not a numeric b-by-z matrix: {path}") from exc
    if values.shape != (len(b), len(z)):
        raise ValueError(f"invalid surface shape in {path}: {values.shape}")
    return SurfaceData(values=values, b=b, z=z, path=path.resolve())


def exact_surface_alignment(left: SurfaceData, right: SurfaceData) -> tuple[bool, str]:
    if left.values.shape != right.values.shape:
        return False, f"shape mismatch {left.values.shape} != {right.values.shape}"
    if not np.array_equal(left.b, right.b):
        return False, "b-grid mismatch"
    if not np.array_equal(left.z, right.z):
        return False, "z-grid mismatch"
    return True, "exact"


def _nested_get(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


COMPARABILITY_FIELDS: tuple[tuple[str, ...], ...] = (
    ("grid", "b_min"),
    ("grid", "b_max"),
    ("grid", "b_points"),
    ("grid", "z_min"),
    ("grid", "z_max"),
    ("grid", "z_points"),
    ("grid", "i_min"),
    ("grid", "i_max"),
    ("grid", "i_points"),
    ("grid", "eta"),
    ("reference_state", "x"),
    ("reference_state", "hatcf"),
    ("reference_state", "lnkf"),
    ("reference_state", "hatc_cal"),
    ("reference_state", "lnk_cal"),
    ("reference_state", "i_low"),
    ("reference_state", "i_mid"),
    ("reference_state", "i_high"),
    ("reference_state", "eta"),
    ("m_mode",),
    ("m_clamp_bounds",),
)


def validate_grid_comparability(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    differences: list[str] = []
    for keys in COMPARABILITY_FIELDS:
        old = _nested_get(previous, keys)
        new = _nested_get(current, keys)
        if old != new:
            differences.append(f"{'.'.join(keys)}: {old!r} != {new!r}")
    return not differences, differences


def read_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        value = pd.read_pickle(path)
    elif path.suffix.lower() == ".csv":
        value = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported dataframe artifact: {path}")
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"artifact is {type(value).__name__}, not pandas.DataFrame: {path}")
    return value


def discover_episode_firm_data(run_root: str | Path) -> tuple[dict[int, Path], list[str]]:
    output_dir = Path(run_root).expanduser().resolve() / "data" / "outputs"
    candidates: dict[int, list[tuple[str, Path]]] = {}
    if output_dir.is_dir():
        for path in output_dir.glob("ep*_stage_*.pkl"):
            if path.stem.endswith("_macro"):
                continue
            match = STAGE_FIRM_RE.match(path.name)
            if match:
                candidates.setdefault(int(match.group("episode")), []).append(
                    (match.group("stage"), path.resolve())
                )
    warnings: list[str] = []
    selected: dict[int, Path] = {}
    stage_priority = {"modeb": 0, "modea": 1, "mode0": 2}
    for episode, values in candidates.items():
        ranked = sorted(values, key=lambda item: (stage_priority.get(item[0], 99), item[0]))
        selected[episode] = ranked[0][1]
        if len(values) > 1:
            warnings.append(
                f"Episode {episode} has multiple firm stage artifacts; selected "
                f"{ranked[0][1].name} using modeb/modea/mode0 priority."
            )
    return dict(sorted(selected.items())), warnings


def select_simulated_state_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select actual current-node firm states without fabricating observations."""
    if "branch" not in frame.columns:
        return frame.copy()
    branch = pd.to_numeric(frame["branch"], errors="coerce")
    if bool((branch == -1).any()):
        return frame.loc[branch == -1].copy()
    if bool((branch == 0).any()):
        return frame.loc[branch == 0].copy()
    return frame.copy()
