"""Vectorized boundary discovery for hard policy regions.

Network evaluations are batched across every active task.  Python loops are
used only across a fixed number of refinement rounds, never across an
economic component, candidate, firm, or bracket.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class BoundarySearchResult:
    boundary_coordinate: torch.Tensor
    boundary_mask: torch.Tensor
    bracket_width: torch.Tensor
    residual: torch.Tensor
    crossing_count: torch.Tensor
    exact_zero_count: torch.Tensor
    nonfinite_count: torch.Tensor


def _rank_within_sorted_rows(row: torch.Tensor) -> torch.Tensor:
    if row.numel() == 0:
        return row.clone()
    change = torch.ones_like(row, dtype=torch.bool)
    change[1:] = row[1:] != row[:-1]
    starts = torch.nonzero(change, as_tuple=False).reshape(-1)
    counts = torch.diff(
        torch.cat((starts, torch.tensor([row.numel()], device=row.device)))
    )
    start_per_item = torch.repeat_interleave(starts, counts)
    return torch.arange(row.numel(), device=row.device) - start_per_item


@torch.inference_mode()
def locate_all_policy_boundaries(
    scan_coordinate: torch.Tensor,
    scan_value: torch.Tensor,
    evaluator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    refinement_rounds: int = 12,
    coordinate_tolerance: float = 1.0e-10,
    zero_tolerance: float = 1.0e-10,
) -> BoundarySearchResult:
    """Locate every visible sign boundary on a common ordered scan.

    Parameters
    ----------
    scan_coordinate, scan_value:
        Matching ``[T,S]`` tensors.  T is the flattened batch of active
        parent/candidate/component tasks.
    evaluator:
        Batched callback receiving ``task_index`` and candidate coordinates.
        One callback is made per refinement round, irrespective of the number
        of active brackets.

    Exact scan zeros are retained as boundaries.  Non-finite segments are not
    relabelled as crossings.  A tangency that does not hit the declared zero
    tolerance is diagnostic-only and is intentionally not invented as a
    boundary.
    """
    if scan_coordinate.shape != scan_value.shape or scan_coordinate.ndim != 2:
        raise ValueError("boundary scan tensors must have matching [T,S] shape")
    if scan_coordinate.shape[1] < 2:
        raise ValueError("boundary scan requires at least two coordinates")
    if not bool((scan_coordinate[:, 1:] > scan_coordinate[:, :-1]).all()):
        raise ValueError("scan coordinates must be strictly increasing")
    if refinement_rounds < 0:
        raise ValueError("refinement_rounds must be non-negative")

    task_count, scan_points = scan_coordinate.shape
    finite = torch.isfinite(scan_value)
    exact = finite & (scan_value.abs() <= float(zero_tolerance))
    crossing = (
        finite[:, :-1]
        & finite[:, 1:]
        & ~exact[:, :-1]
        & ~exact[:, 1:]
        & (torch.signbit(scan_value[:, :-1]) != torch.signbit(scan_value[:, 1:]))
    )
    pair = torch.nonzero(crossing, as_tuple=False)
    if pair.numel():
        task = pair[:, 0]
        column = pair[:, 1]
        left = scan_coordinate[task, column].clone()
        right = scan_coordinate[task, column + 1].clone()
        f_left = scan_value[task, column].clone()
        f_right = scan_value[task, column + 1].clone()
        for _ in range(int(refinement_rounds)):
            active = (right - left) > float(coordinate_tolerance)
            if not bool(active.any()):
                break
            active_position = torch.nonzero(active, as_tuple=False).reshape(-1)
            midpoint = 0.5 * (left[active_position] + right[active_position])
            f_mid = evaluator(task[active_position], midpoint)
            finite_mid = torch.isfinite(f_mid)
            # Non-finite midpoint values stop only their own bracket; they are
            # preserved in diagnostics instead of creating a false boundary.
            usable_position = active_position[finite_mid]
            if usable_position.numel() == 0:
                break
            usable_mid = f_mid[finite_mid]
            same_left = torch.signbit(usable_mid) == torch.signbit(f_left[usable_position])
            left[usable_position] = torch.where(
                same_left, midpoint[finite_mid], left[usable_position]
            )
            f_left[usable_position] = torch.where(
                same_left, usable_mid, f_left[usable_position]
            )
            right[usable_position] = torch.where(
                same_left, right[usable_position], midpoint[finite_mid]
            )
            f_right[usable_position] = torch.where(
                same_left, f_right[usable_position], usable_mid
            )
        crossing_coordinate = 0.5 * (left + right)
        crossing_width = right - left
        choose_left = f_left.abs() <= f_right.abs()
        crossing_residual = torch.where(choose_left, f_left.abs(), f_right.abs())
    else:
        task = torch.empty(0, device=scan_value.device, dtype=torch.long)
        crossing_coordinate = scan_value.new_empty((0,))
        crossing_width = scan_value.new_empty((0,))
        crossing_residual = scan_value.new_empty((0,))

    exact_pos = torch.nonzero(exact, as_tuple=False)
    exact_task = exact_pos[:, 0] if exact_pos.numel() else task.new_empty((0,))
    exact_coordinate = (
        scan_coordinate[exact_pos[:, 0], exact_pos[:, 1]]
        if exact_pos.numel()
        else scan_value.new_empty((0,))
    )
    all_task = torch.cat((task, exact_task))
    all_coordinate = torch.cat((crossing_coordinate, exact_coordinate))
    all_width = torch.cat((crossing_width, torch.zeros_like(exact_coordinate)))
    all_residual = torch.cat((crossing_residual, torch.zeros_like(exact_coordinate)))

    capacity = 2 * scan_points
    boundary = torch.full(
        (task_count, capacity), float("inf"), device=scan_value.device, dtype=scan_value.dtype
    )
    width = torch.full_like(boundary, float("nan"))
    residual = torch.full_like(boundary, float("nan"))
    if all_task.numel():
        order = torch.argsort(all_task * (all_task.numel() + 1) + torch.arange(all_task.numel(), device=all_task.device))
        sorted_task = all_task[order]
        rank = _rank_within_sorted_rows(sorted_task)
        boundary[sorted_task, rank] = all_coordinate[order]
        width[sorted_task, rank] = all_width[order]
        residual[sorted_task, rank] = all_residual[order]
    boundary, sort_order = boundary.sort(dim=1)
    width = torch.gather(width, 1, sort_order)
    residual = torch.gather(residual, 1, sort_order)
    mask = torch.isfinite(boundary)
    if capacity > 1:
        duplicate = mask[:, 1:] & mask[:, :-1] & (
            (boundary[:, 1:] - boundary[:, :-1]).abs()
            <= max(float(coordinate_tolerance) * 2.0, float(zero_tolerance))
        )
        mask[:, 1:] &= ~duplicate
    compact_rank = mask.to(torch.long).cumsum(dim=1) - 1
    compact_boundary = torch.full_like(boundary, float("nan"))
    compact_width = torch.full_like(width, float("nan"))
    compact_residual = torch.full_like(residual, float("nan"))
    row, col = torch.nonzero(mask, as_tuple=True)
    if row.numel():
        destination = compact_rank[row, col]
        compact_boundary[row, destination] = boundary[row, col]
        compact_width[row, destination] = width[row, col]
        compact_residual[row, destination] = residual[row, col]
    compact_mask = torch.isfinite(compact_boundary)
    return BoundarySearchResult(
        boundary_coordinate=compact_boundary,
        boundary_mask=compact_mask,
        bracket_width=compact_width,
        residual=compact_residual,
        crossing_count=crossing.sum(dim=1),
        exact_zero_count=exact.sum(dim=1),
        nonfinite_count=(~finite).sum(dim=1),
    )


def interval_bounds_from_boundaries(
    boundary_coordinate: torch.Tensor,
    boundary_mask: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create padded disjoint intervals spanning ``[lower,upper]``."""
    if boundary_coordinate.shape != boundary_mask.shape:
        raise ValueError("boundary coordinate and mask must match")
    if not lower < upper:
        raise ValueError("invalid interval support")
    task_count, capacity = boundary_coordinate.shape
    count = boundary_mask.sum(dim=1)
    left = torch.full(
        (task_count, capacity + 1), float(lower),
        device=boundary_coordinate.device, dtype=boundary_coordinate.dtype,
    )
    right = torch.full_like(left, float(upper))
    safe = torch.where(boundary_mask, boundary_coordinate, torch.full_like(boundary_coordinate, float(upper)))
    left[:, 1:] = safe
    right[:, :-1] = safe
    position = torch.arange(capacity + 1, device=boundary_coordinate.device)[None, :]
    interval_mask = position <= count[:, None]
    interval_mask &= right > left
    return left, right, interval_mask

