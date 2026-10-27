"""Deterministic near-tie resolution for the Grid BP candidate argmax.

Regression suite for the ``decisions_bp`` production-gate failure: plain
``argmax`` flipped between near-equal leverage candidates when the same
objective grid was evaluated twice with float-rounding-scale differences
(batch-reduction order / FP64-vs-FP32 replay).  The fix treats candidates
within a relative tolerance of the row maximum as tied and always picks the
lowest candidate index.
"""
import torch

from training.bp_grid_teacher import _deterministic_argmax


def test_exact_tie_picks_lowest_index():
    grid = torch.tensor([[1.0, 3.0, 3.0, 2.0]])
    out = _deterministic_argmax(grid)
    assert out.shape == (1, 1)
    assert out.dtype == torch.long
    assert int(out[0, 0]) == 1  # index 1 and 2 tie at 3.0 -> lowest wins


def test_near_tie_within_rtol_picks_lowest_index():
    # gap 1e-9 << rtol=1e-6 -> treated as tie
    grid = torch.tensor([[0.0, 1.0, 1.0 - 1e-9, 0.5]])
    assert int(_deterministic_argmax(grid)[0, 0]) == 1


def test_distinct_gap_unchanged():
    # genuine separation (1e-3) must still pick the true argmax
    grid = torch.tensor([[0.0, 1.0, 1.0 - 1e-3, 0.5]])
    assert int(_deterministic_argmax(grid)[0, 0]) == 1
    grid = torch.tensor([[0.0, 1.0 - 1e-3, 1.0, 0.5]])
    assert int(_deterministic_argmax(grid)[0, 0]) == 2


def test_jitter_stability_no_flip():
    # reproduces the production-gate failure mode: two evaluations of the
    # same objective grid with ~1e-7 float jitter must not flip the choice
    base = torch.tensor([[0.2, 0.5, 0.5 + 5e-8, 0.1]], dtype=torch.float32)
    jitter = torch.tensor([[1e-8, -1e-8, 1e-8, -1e-8]], dtype=torch.float32)
    a = _deterministic_argmax(base + jitter)
    b = _deterministic_argmax(base - jitter)
    # candidates 1 and 2 are within the tie tolerance of the row max; 0.2 is
    # not. Both evaluations must pick the same tied member (lowest index = 1)
    assert int(a[0, 0]) == int(b[0, 0]) == 1


def test_rows_independent():
    grid = torch.tensor(
        [
            [1.0, 2.0, 2.0, 0.0],  # tie at 1,2 -> 1
            [5.0, 4.0, 4.0 + 1e-12, 0.0],  # unique max at 0 -> 0
            [0.0, 0.0, 0.0, 0.0],  # all tie -> 0
        ]
    )
    out = _deterministic_argmax(grid).reshape(-1).tolist()
    assert out == [1, 0, 0]


def test_large_value_scale_relative_tol():
    # vmax=1e6 -> tol = 1e-6 * 1e6 = 1.0; gap 0.5 is a tie, gap 2.0 is not
    grid = torch.tensor([[1e6, 1e6 - 0.5, 0.0]])
    assert int(_deterministic_argmax(grid)[0, 0]) == 0
    grid = torch.tensor([[1e6 - 2.0, 1e6, 0.0]])
    assert int(_deterministic_argmax(grid)[0, 0]) == 1


def test_nan_row_falls_back_without_crash():
    grid = torch.tensor([[float("nan"), float("nan"), float("nan")]])
    out = _deterministic_argmax(grid)
    assert out.shape == (1, 1)
    assert int(out[0, 0]) in (0, 1, 2)


def test_empty_tie_mask_falls_back():
    # negative-infinity row: no candidate within tolerance of vmax via >=
    grid = torch.tensor([[float("-inf"), float("-inf")]])
    out = _deterministic_argmax(grid)
    assert out.shape == (1, 1)
