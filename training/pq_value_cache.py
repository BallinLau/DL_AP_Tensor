"""Cached P/Q value targets for staged Policy/Value training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class PQValueTargetBatch:
    batch_id: int
    p0_value_target: torch.Tensor
    pi_value_target: torch.Tensor
    teacher_hash: str
    parent_hash: str
    children_hash: str
    m_hash: str
    grid_config_hash: str
    source_id: Optional[torch.Tensor] = None
    source_index: Optional[torch.Tensor] = None
