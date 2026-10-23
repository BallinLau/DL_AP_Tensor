"""
Tensor-first data containers for simulation outputs.

These containers keep data on device during simulation/training and only
materialize pandas DataFrames at the very end when explicitly requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch


@dataclass
class TensorTable:
    """
    A 2D tensor table with explicit column names.
    """

    data: torch.Tensor
    columns: Sequence[str]

    def __post_init__(self) -> None:
        if self.data.dim() != 2:
            raise ValueError(f"TensorTable expects 2D tensor, got shape={tuple(self.data.shape)}")
        if self.data.shape[1] != len(self.columns):
            raise ValueError(
                f"Column mismatch: data has {self.data.shape[1]} cols, "
                f"columns has {len(self.columns)}"
            )

    def to(self, device: torch.device) -> "TensorTable":
        return TensorTable(data=self.data.to(device), columns=list(self.columns))

    def to_dataframe(self) -> pd.DataFrame:
        arr = self.data.detach().cpu().numpy()
        return pd.DataFrame(arr, columns=list(self.columns))

    def __len__(self) -> int:
        return int(self.data.shape[0])


@dataclass
class TensorSimulationOutput:
    """
    Unified tensor output for firm-level and macro-level panels.
    """

    firm: TensorTable
    macro: TensorTable
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.firm.to_dataframe(), self.macro.to_dataframe()


def cat_rows(rows: Iterable[torch.Tensor], n_cols: int, device: torch.device) -> torch.Tensor:
    """
    Concatenate row tensors safely. Returns an empty tensor when no rows are present.

    Rows are copied into a preallocated host buffer and freed as they are
    consumed: a plain ``torch.cat`` keeps every input and the concatenated
    output resident at the same time (2x peak) and OOMs on late-simulation
    pools. When the caller passes a list, it is drained first so its references
    do not keep the source rows alive. The result is always returned on
    ``device``.
    """
    rows_list: List[torch.Tensor] = [r for r in rows if r is not None and r.numel() > 0]
    if not rows_list:
        return torch.empty((0, n_cols), device=device, dtype=torch.float32)
    if isinstance(rows, list):
        rows.clear()
    n_rows = sum(int(r.shape[0]) for r in rows_list)
    out_cpu = torch.empty((n_rows, n_cols), dtype=rows_list[0].dtype, device="cpu")
    offset = 0
    for idx in range(len(rows_list)):
        row = rows_list[idx]
        n = int(row.shape[0])
        out_cpu[offset:offset + n].copy_(row)
        offset += n
        rows_list[idx] = None
    return out_cpu.to(device)
