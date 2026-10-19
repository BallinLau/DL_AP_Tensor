from __future__ import annotations

import pytest
import torch

from data.batched_policy_boundaries import (
    interval_bounds_from_boundaries,
    locate_all_policy_boundaries,
)


def test_all_visible_boundaries_are_refined_in_one_active_batch():
    grid = torch.linspace(-2.0, 2.0, 17, dtype=torch.float64)
    scan = grid[None, :].expand(2, -1)

    def value(task, x):
        first = (x + 1.0) * (x - 0.25) * (x - 1.25)
        second = x - 0.5
        return torch.where(task == 0, first, second)

    task = torch.arange(2)[:, None].expand_as(scan)
    result = locate_all_policy_boundaries(
        scan,
        value(task, scan),
        value,
        refinement_rounds=40,
        coordinate_tolerance=1e-12,
        zero_tolerance=1e-12,
    )
    roots0 = result.boundary_coordinate[0, result.boundary_mask[0]]
    roots1 = result.boundary_coordinate[1, result.boundary_mask[1]]
    torch.testing.assert_close(
        roots0, torch.tensor([-1.0, 0.25, 1.25], dtype=torch.float64), atol=1e-9, rtol=0
    )
    torch.testing.assert_close(
        roots1, torch.tensor([0.5], dtype=torch.float64), atol=1e-9, rtol=0
    )

    left, right, mask = interval_bounds_from_boundaries(
        result.boundary_coordinate, result.boundary_mask, lower=-2.0, upper=2.0
    )
    assert mask[0].sum().item() == 4
    assert mask[1].sum().item() == 2
    # The interval tensor carries only the maximum number of observed
    # boundaries, not the conservative 2 * scan_points search capacity.
    assert left.shape == right.shape == mask.shape == (2, 4)
    assert left[0, 0].item() == pytest.approx(-2.0)
    assert right[0, 3].item() == pytest.approx(2.0)


def test_interval_padding_is_trimmed_without_changing_empty_rows():
    boundary = torch.tensor(
        [[0.25, float("nan"), float("nan"), float("nan")],
         [float("nan"), float("nan"), float("nan"), float("nan")]],
        dtype=torch.float64,
    )
    boundary_mask = torch.isfinite(boundary)
    left, right, mask = interval_bounds_from_boundaries(
        boundary, boundary_mask, lower=0.0, upper=1.0
    )
    assert left.shape == right.shape == mask.shape == (2, 2)
    torch.testing.assert_close(left[0], torch.tensor([0.0, 0.25], dtype=torch.float64))
    torch.testing.assert_close(right[0], torch.tensor([0.25, 1.0], dtype=torch.float64))
    assert mask[0].tolist() == [True, True]
    assert mask[1].tolist() == [True, False]


def test_nonfinite_segment_does_not_create_false_boundary():
    scan = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)[None, :]
    values = scan.clone()
    values[:, 4] = float("nan")

    def evaluator(task, x):
        return x

    result = locate_all_policy_boundaries(scan, values, evaluator)
    assert result.boundary_mask.sum().item() == 0
    assert result.nonfinite_count.item() == 1


def test_exact_zero_is_retained_without_duplicate_crossings():
    scan = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)[None, :]
    result = locate_all_policy_boundaries(
        scan, scan.clone(), lambda task, x: x, zero_tolerance=1e-14
    )
    roots = result.boundary_coordinate[result.boundary_mask]
    assert roots.numel() == 1
    assert roots.item() == pytest.approx(0.0)

