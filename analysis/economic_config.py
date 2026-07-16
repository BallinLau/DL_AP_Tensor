from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from config import Config


@dataclass(frozen=True)
class AnalysisEconomicConfig:
    RHO_X: float
    SIGMA_X: float
    XBAR: float
    RHO_Z: float
    SIGMA_Z: float
    ZBAR: float
    ZETA: float
    I_THRESHOLD: float
    G: float
    DELTA: float
    PHI: float
    TAU: float
    KAPPA_B: float
    KAPPA_E: float
    AIO_WEIGHT: float
    ALPHA_Z: float
    BETA_Z: float
    Z0: float

    @classmethod
    def from_current_config(cls) -> "AnalysisEconomicConfig":
        return cls.from_dict({name: getattr(Config, name) for name in cls.field_names()})

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(cls.__dataclass_fields__.keys())

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "AnalysisEconomicConfig":
        missing = [name for name in cls.field_names() if name not in values]
        if missing:
            raise ValueError(f"economic config is missing required fields: {missing}")
        return cls(**{name: float(values[name]) for name in cls.field_names()})

    @classmethod
    def from_json(cls, path: str | Path) -> "AnalysisEconomicConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "config_snapshot" in payload:
            payload = payload["config_snapshot"]
        if not isinstance(payload, dict):
            raise ValueError(f"config_json must contain a JSON object: {path}")
        return cls.from_dict(payload)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
