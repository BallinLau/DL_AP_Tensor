from __future__ import annotations

import pytest

from experiments.analyze_inner_fp_block_isolation import _trend


def test_trend_labels_monotone_improvement_without_pass_threshold():
    label, change, slope = _trend([5.0, 4.0, 3.0, 2.0, 1.0])
    assert label == "improving"
    assert change == pytest.approx(-4.0)
    assert slope < 0.0


def test_trend_labels_monotone_divergence_without_pass_threshold():
    label, change, slope = _trend([1.0, 2.0, 3.0, 4.0, 5.0])
    assert label == "diverging"
    assert change == pytest.approx(4.0)
    assert slope > 0.0


def test_trend_marks_mixed_path_as_stalled_or_mixed():
    label, _, _ = _trend([1.0, 4.0, 3.0, 2.0, 1.5])
    assert label == "flat/stalled or mixed"
