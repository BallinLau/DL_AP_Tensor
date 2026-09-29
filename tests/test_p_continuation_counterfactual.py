import json

import numpy as np
import pandas as pd
import pytest
import torch

from evaluation.full_run_diagnostics import model_state_hash
from experiments.evaluate_p_continuation_counterfactual import (
    build_cube,
    build_shapley_tables,
    contribution_ratio,
    cube_column,
    existing_analysis_child_shocks,
    fixed_bank_shapley,
    require_finite,
    three_factor_shapley,
    weighted_continuation,
)


OLD = 1
NEW = 4


def _cube(**overrides):
    values = {
        cube_column(m, p, r): 0.0
        for m in (OLD, NEW)
        for p in (OLD, NEW)
        for r in (OLD, NEW)
    }
    values.update(overrides)
    return values


def test_cube_indexing_uses_full_m_p_r_names():
    assert cube_column(1, 4, 1) == "F_M1_P4_R1"
    frames = {
        1: pd.DataFrame({
            "b": [0.05], "z": [4.0], "eta": [1.0],
            "F_M1_P1": [1.0], "F_M1_P4": [2.0],
            "F_M4_P1": [3.0], "F_M4_P4": [4.0],
        }),
        4: pd.DataFrame({
            "b": [0.05], "z": [4.0], "eta": [1.0],
            "F_M1_P1": [5.0], "F_M1_P4": [6.0],
            "F_M4_P1": [7.0], "F_M4_P4": [8.0],
        }),
    }
    cube = build_cube(frames, old_episode=OLD, new_episode=NEW)
    assert cube.loc[0, "F_M1_P1_R1"] == 1.0
    assert cube.loc[0, "F_M4_P4_R4"] == 8.0


def test_three_factor_shapley_identity_with_interactions():
    values = _cube(
        F_M1_P1_R1=1.0,
        F_M4_P1_R1=2.5,
        F_M1_P4_R1=4.0,
        F_M1_P1_R4=-1.0,
        F_M4_P4_R1=8.0,
        F_M4_P1_R4=5.5,
        F_M1_P4_R4=7.0,
        F_M4_P4_R4=13.0,
    )
    result = three_factor_shapley(values, old_episode=OLD, new_episode=NEW)
    assert result["shapley_identity_error"] == pytest.approx(0.0, abs=1e-12)
    assert sum(
        result[name]
        for name in ("future_equity_effect", "sdf_effect", "transition_policy_effect")
    ) == pytest.approx(12.0)


@pytest.mark.parametrize(
    ("factor", "expected"),
    [("P", (2.0, 0.0, 0.0)), ("M", (0.0, 2.0, 0.0)), ("R", (0.0, 0.0, 2.0))],
)
def test_single_factor_change_is_fully_assigned(factor, expected):
    values = {}
    for m in (OLD, NEW):
        for p in (OLD, NEW):
            for r in (OLD, NEW):
                active = {"M": m == NEW, "P": p == NEW, "R": r == NEW}[factor]
                values[cube_column(m, p, r)] = 2.0 if active else 0.0
    result = three_factor_shapley(values, old_episode=OLD, new_episode=NEW)
    actual = (
        result["future_equity_effect"],
        result["sdf_effect"],
        result["transition_policy_effect"],
    )
    assert actual == pytest.approx(expected)


def test_fixed_bank_two_factor_shapley_identity():
    values = _cube(
        F_M1_P1_R1=1.0,
        F_M1_P4_R1=4.0,
        F_M4_P1_R1=3.0,
        F_M4_P4_R1=9.0,
    )
    result = fixed_bank_shapley(
        values, old_episode=OLD, new_episode=NEW, bank_episode=OLD
    )
    assert result["future_equity_effect"] + result["sdf_effect"] == pytest.approx(8.0)
    assert result["identity_error"] == pytest.approx(0.0)


def test_nan_is_not_silently_replaced():
    with pytest.raises(ValueError, match="nonfinite"):
        require_finite("cube", [1.0, np.nan])
    with pytest.raises(ValueError, match="nonfinite"):
        weighted_continuation(
            torch.tensor([[1.0, float("nan")]]),
            torch.ones(1, 2),
            torch.full((1, 2), 0.5),
            torch.ones(1),
        )


def test_ratios_keep_negative_and_greater_than_one_values():
    assert contribution_ratio(-2.0, 1.0) == -2.0
    assert contribution_ratio(3.0, 2.0) == 1.5
    assert np.isnan(contribution_ratio(1.0, 0.0))


def test_pi_multiplier_is_applied_once():
    result = weighted_continuation(
        torch.tensor([[2.0, 2.0]]),
        torch.tensor([[3.0, 3.0]]),
        torch.tensor([[0.5, 0.5]]),
        torch.tensor([1.14]),
    )
    assert result.item() == pytest.approx(1.14 * 6.0)
    assert result.item() != pytest.approx((1.14 ** 2) * 6.0)


def test_pchild_is_used_directly_without_an_extra_survival_gate():
    # Pchild is already production P=max(Phat,0). A separate survival value must
    # not be multiplied into the continuation operator.
    result = weighted_continuation(
        torch.tensor([[3.0]]),
        torch.tensor([[2.0]]),
        torch.tensor([[1.0]]),
        torch.tensor([1.0]),
    )
    assert result.item() == pytest.approx(6.0)


def test_build_shapley_tables_rejects_nonfinite_cube():
    row = {"b": 0.05, "z": 4.0, "eta": 1.0, **_cube()}
    row["F_M4_P4_R4"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        build_shapley_tables(
            pd.DataFrame([row]), old_episode=OLD, new_episode=NEW
        )


def test_read_only_forward_preserves_model_hash():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1))
    before = model_state_hash(model)
    model.eval()
    with torch.no_grad():
        model(torch.ones(4, 2))
    assert model_state_hash(model) == before


def test_existing_analysis_child_count_is_explicit(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"n_child_shocks": 64}), encoding="utf-8"
    )
    assert existing_analysis_child_shocks(tmp_path) == 64
    assert existing_analysis_child_shocks(tmp_path / "missing") is None
