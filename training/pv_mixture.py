"""Policy/Value parent-group mixture sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
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


def resolve_val_count(total: int, val_fraction: float) -> int:
    if total <= 1:
        return 0
    val_fraction = min(max(float(val_fraction), 0.0), 0.5)
    if val_fraction <= 0.0:
        return 0
    n_val = max(1, int(round(total * val_fraction)))
    return min(n_val, total - 1)


def allocate_source_val_counts(
    source_sizes: List[int],
    val_fraction: float,
) -> List[int]:
    total = int(sum(source_sizes))
    target_total = resolve_val_count(total, val_fraction)
    if target_total <= 0:
        return [0 for _ in source_sizes]

    ideals = [float(size) * float(val_fraction) for size in source_sizes]
    counts = [
        min(int(math.floor(ideal)), max(int(size) - 1, 0))
        for size, ideal in zip(source_sizes, ideals)
    ]
    remaining = target_total - int(sum(counts))
    order = sorted(
        range(len(source_sizes)),
        key=lambda idx: (ideals[idx] - math.floor(ideals[idx]), source_sizes[idx]),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for idx in order:
            cap = max(int(source_sizes[idx]) - 1, 0)
            if counts[idx] < cap:
                counts[idx] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    while remaining < 0:
        progressed = False
        for idx in reversed(order):
            if counts[idx] > 0:
                counts[idx] -= 1
                remaining += 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return counts


def split_pool_with_val_count(
    pool: PVParentGroupPool,
    n_val: int,
    generator: torch.Generator,
) -> Tuple[PVParentGroupPool, PVParentGroupPool]:
    n_val = min(max(int(n_val), 0), max(len(pool) - 1, 0))
    if len(pool) == 0:
        return pool, pool
    if n_val == 0:
        return pool, select_parent_groups(pool, torch.empty(0, dtype=torch.long))
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
        sim_val_count, cov_val_count = allocate_source_val_counts(
            [len(sim_selected), len(coverage_selected)],
            val_fraction,
        )
        sim_train, sim_val = split_pool_with_val_count(sim_selected, sim_val_count, generator)
        cov_train, cov_val = split_pool_with_val_count(coverage_selected, cov_val_count, generator)
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
        "global_validation_target_parent_groups": float(resolve_val_count(n_sim + n_coverage, val_fraction)),
        "allocated_sim_validation_parent_groups": float(sim_val_count if stratified_validation else val_counts["source0"]),
        "allocated_coverage_validation_parent_groups": float(cov_val_count if stratified_validation else val_counts["source1"]),
        "actual_coverage_ratio": float(n_coverage / max(n_sim + n_coverage, 1)),
        "train_coverage_ratio": float(train_counts["source1"] / max(len(train_pool), 1)),
        "validation_coverage_ratio": float(val_counts["source1"] / max(len(val_pool), 1)),
    }
    return train_pool, val_pool, summary


def build_selected_mixture_split(
    sim_selected: PVParentGroupPool,
    coverage_selected: PVParentGroupPool,
    val_fraction: float,
    generator: torch.Generator,
    stratified_validation: bool = True,
    *,
    sim_parent_groups_available: int,
    coverage_parent_groups_available: int,
    total_parent_budget: int,
) -> Tuple[PVParentGroupPool, PVParentGroupPool, Dict[str, float]]:
    if stratified_validation:
        sim_val_count, cov_val_count = allocate_source_val_counts(
            [len(sim_selected), len(coverage_selected)],
            val_fraction,
        )
        sim_train, sim_val = split_pool_with_val_count(sim_selected, sim_val_count, generator)
        cov_train, cov_val = split_pool_with_val_count(coverage_selected, cov_val_count, generator)
        train_pool = concat_parent_group_pools([sim_train, cov_train])
        val_pools = [pool for pool in [sim_val, cov_val] if len(pool) > 0]
        val_pool = (
            concat_parent_group_pools(val_pools)
            if val_pools
            else select_parent_groups(train_pool, torch.empty(0, dtype=torch.long))
        )
    else:
        combined = concat_parent_group_pools([sim_selected, coverage_selected])
        train_pool, val_pool = split_pool(combined, val_fraction, generator)

    train_counts = source_counts(train_pool)
    val_counts = source_counts(val_pool)
    n_sim = len(sim_selected)
    n_coverage = len(coverage_selected)
    summary: Dict[str, float] = {
        "sim_parent_groups_available": float(sim_parent_groups_available),
        "coverage_parent_groups_available": float(coverage_parent_groups_available),
        "total_parent_budget": float(total_parent_budget),
        "sim_parent_groups_selected": float(n_sim),
        "coverage_parent_groups_selected": float(n_coverage),
        "mixed_parent_groups_selected": float(n_sim + n_coverage),
        "train_parent_groups": float(len(train_pool)),
        "validation_parent_groups": float(len(val_pool)),
        "train_sim_parent_groups": float(train_counts["source0"]),
        "train_coverage_parent_groups": float(train_counts["source1"]),
        "validation_sim_parent_groups": float(val_counts["source0"]),
        "validation_coverage_parent_groups": float(val_counts["source1"]),
        "global_validation_target_parent_groups": float(resolve_val_count(n_sim + n_coverage, val_fraction)),
        "allocated_sim_validation_parent_groups": float(sim_val_count if stratified_validation else val_counts["source0"]),
        "allocated_coverage_validation_parent_groups": float(cov_val_count if stratified_validation else val_counts["source1"]),
        "actual_coverage_ratio": float(n_coverage / max(n_sim + n_coverage, 1)),
        "train_coverage_ratio": float(train_counts["source1"] / max(len(train_pool), 1)),
        "validation_coverage_ratio": float(val_counts["source1"] / max(len(val_pool), 1)),
    }
    return train_pool, val_pool, summary
