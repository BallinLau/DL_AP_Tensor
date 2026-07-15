"""Policy/Value parent-group mixture sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class PVParentGroupPool:
    """Aligned parent/children tensors plus non-model source metadata."""

    parent: torch.Tensor
    children: List[torch.Tensor]
    source_id: torch.Tensor
    source_index: torch.Tensor

    def __len__(self) -> int:
        return int(self.parent.shape[0])


def _index_on_pool_device(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    return indices.to(device=device, dtype=torch.long)


def select_parent_groups(pool: PVParentGroupPool, indices: torch.Tensor) -> PVParentGroupPool:
    idx = _index_on_pool_device(indices, pool.parent.device)
    return PVParentGroupPool(
        parent=pool.parent[idx],
        children=[child[idx] for child in pool.children],
        source_id=pool.source_id[idx],
        source_index=pool.source_index[idx],
    )


def concat_parent_group_pools(pools: List[PVParentGroupPool]) -> PVParentGroupPool:
    non_empty = [pool for pool in pools if len(pool) > 0]
    if not non_empty:
        raise ValueError("Cannot concatenate an empty list of PV parent-group pools.")
    n_children = len(non_empty[0].children)
    for pool in non_empty:
        if len(pool.children) != n_children:
            raise ValueError("All PV parent-group pools must have the same number of child branches.")
    return PVParentGroupPool(
        parent=torch.cat([pool.parent for pool in non_empty], dim=0),
        children=[
            torch.cat([pool.children[k] for pool in non_empty], dim=0)
            for k in range(n_children)
        ],
        source_id=torch.cat([pool.source_id for pool in non_empty], dim=0),
        source_index=torch.cat([pool.source_index for pool in non_empty], dim=0),
    )


def source_counts(pool: PVParentGroupPool) -> Dict[str, int]:
    if len(pool) == 0:
        return {"source0": 0, "source1": 0}
    sid = pool.source_id.detach().cpu().to(torch.long)
    return {
        "source0": int((sid == 0).sum().item()),
        "source1": int((sid == 1).sum().item()),
    }


def fixed_total_source_counts(total: int, coverage_ratio: float) -> Tuple[int, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    ratio = min(max(float(coverage_ratio), 0.0), 1.0)
    n_coverage = int(round(total * ratio))
    n_coverage = min(max(n_coverage, 0), total)
    return total - n_coverage, n_coverage


def sample_pool_without_replacement(
    pool: PVParentGroupPool,
    n_keep: int,
    generator: torch.Generator,
) -> PVParentGroupPool:
    n_keep = int(n_keep)
    if n_keep < 0:
        raise ValueError("n_keep must be non-negative")
    if n_keep > len(pool):
        raise ValueError(f"Requested {n_keep} parent groups from a pool with {len(pool)} groups.")
    if n_keep == len(pool):
        return pool
    perm = torch.randperm(len(pool), generator=generator)[:n_keep]
    return select_parent_groups(pool, perm)


def split_pool(
    pool: PVParentGroupPool,
    val_fraction: float,
    generator: torch.Generator,
) -> Tuple[PVParentGroupPool, PVParentGroupPool]:
    if len(pool) <= 1:
        return pool, select_parent_groups(pool, torch.empty(0, dtype=torch.long))
    val_fraction = min(max(float(val_fraction), 0.0), 0.5)
    n_val = int(round(len(pool) * val_fraction))
    if val_fraction > 0.0:
        n_val = max(1, n_val)
    n_val = min(n_val, len(pool) - 1)
    perm = torch.randperm(len(pool), generator=generator)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return select_parent_groups(pool, train_idx), select_parent_groups(pool, val_idx)


def build_fixed_total_mixture_split(
    sim_pool: PVParentGroupPool,
    coverage_pool: PVParentGroupPool,
    coverage_ratio: float,
    val_fraction: float,
    generator: torch.Generator,
    stratified_validation: bool = True,
) -> Tuple[PVParentGroupPool, PVParentGroupPool, Dict[str, float]]:
    n_total = len(sim_pool)
    n_sim, n_coverage = fixed_total_source_counts(n_total, coverage_ratio)
    if n_coverage > len(coverage_pool):
        raise ValueError(
            f"Coverage pool too small: requested {n_coverage}, available {len(coverage_pool)}."
        )
    sim_selected = sample_pool_without_replacement(sim_pool, n_sim, generator)
    coverage_selected = sample_pool_without_replacement(coverage_pool, n_coverage, generator)

    if stratified_validation:
        sim_train, sim_val = split_pool(sim_selected, val_fraction, generator)
        cov_train, cov_val = split_pool(coverage_selected, val_fraction, generator)
        train_pool = concat_parent_group_pools([sim_train, cov_train])
        val_pools = [pool for pool in [sim_val, cov_val] if len(pool) > 0]
        val_pool = concat_parent_group_pools(val_pools) if val_pools else select_parent_groups(train_pool, torch.empty(0, dtype=torch.long))
    else:
        combined = concat_parent_group_pools([sim_selected, coverage_selected])
        train_pool, val_pool = split_pool(combined, val_fraction, generator)

    train_counts = source_counts(train_pool)
    val_counts = source_counts(val_pool)
    summary: Dict[str, float] = {
        "sim_parent_groups_available": float(len(sim_pool)),
        "coverage_parent_groups_available": float(len(coverage_pool)),
        "sim_parent_groups_selected": float(n_sim),
        "coverage_parent_groups_selected": float(n_coverage),
        "mixed_parent_groups_selected": float(n_sim + n_coverage),
        "train_parent_groups": float(len(train_pool)),
        "validation_parent_groups": float(len(val_pool)),
        "train_sim_parent_groups": float(train_counts["source0"]),
        "train_coverage_parent_groups": float(train_counts["source1"]),
        "validation_sim_parent_groups": float(val_counts["source0"]),
        "validation_coverage_parent_groups": float(val_counts["source1"]),
        "actual_coverage_ratio": float(n_coverage / max(n_sim + n_coverage, 1)),
    }
    return train_pool, val_pool, summary
