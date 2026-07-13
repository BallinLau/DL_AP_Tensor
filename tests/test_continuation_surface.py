from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch

from config import Config
from experiments.export_continuation_surface import (
    compute_issue_bar_i,
    make_default_boundary_frame,
    make_surface_states,
    validate_q_decomposition_frame,
)
from losses.q_loss import QLoss, compute_q_survival_recovery_components


def test_make_surface_states_varies_only_child_b_and_z():
    child = torch.tensor([[0.2, 0.1, 1.0, 0.3, 0.4, 0.5, 0.6]], dtype=torch.float64)
    states, b, z = make_surface_states(child, b_grid_size=3, z_min=-1.0, z_max=1.0, z_grid_size=5)
    assert states.shape == (15, 7)
    assert b.shape == (15, 1)
    assert z.shape == (15, 1)
    torch.testing.assert_close(states[:, 0:1], b)
    torch.testing.assert_close(states[:, 1:2], z)
    torch.testing.assert_close(states[:, 2:], child.expand(15, -1)[:, 2:])


def test_q_survival_recovery_decomposition_identity():
    m = torch.tensor([[0.9], [1.1]], dtype=torch.float64)
    b = torch.tensor([[0.2], [0.5]], dtype=torch.float64)
    q_child = torch.tensor([[0.1], [0.2]], dtype=torch.float64)
    default = torch.tensor([[0.0], [0.7]], dtype=torch.float64)
    x = torch.tensor([[0.3], [0.4]], dtype=torch.float64)
    z = torch.tensor([[0.1], [0.2]], dtype=torch.float64)
    multiplier = Config.G
    q_issue = torch.tensor([[0.25], [0.35]], dtype=torch.float64)
    recovery = b * Config.PHI * (1.0 - Config.DELTA + torch.exp(x + z))
    survival_part = m * (b + multiplier * q_child) * (1.0 - default)
    recovery_part = m * recovery * multiplier * default
    total = survival_part + recovery_part
    residual = total - q_issue
    torch.testing.assert_close(residual, total - q_issue)
    torch.testing.assert_close(q_issue - total, -residual)
    recovery_share = recovery_part / total.clamp_min(1e-12)
    assert torch.isfinite(recovery_share).all()


def test_q_decomposition_helper_matches_q_loss_residual():
    loss_fn = QLoss()
    Q = torch.tensor([[0.4], [0.6]], dtype=torch.float64)
    b = torch.tensor([[0.3], [0.7]], dtype=torch.float64)
    bar_i = torch.tensor([[0.2], [0.8]], dtype=torch.float64)
    M = torch.tensor([[0.9], [1.1]], dtype=torch.float64)
    Qsp = torch.tensor([[0.5], [0.2]], dtype=torch.float64)
    bar_z = torch.tensor([[0.1], [0.6]], dtype=torch.float64)
    x = torch.tensor([[0.2], [0.3]], dtype=torch.float64)
    z = torch.tensor([[0.4], [0.5]], dtype=torch.float64)
    residual = loss_fn.compute_main_residual(
        Q,
        b,
        bar_i,
        M_list=[M],
        Qsp_children=[Qsp],
        bar_z_children=[bar_z],
        x_children=[x],
        z_children=[z],
    )[0]
    components = compute_q_survival_recovery_components(
        Q=Q,
        b=b,
        bar_i=bar_i,
        M=M,
        Qsp=Qsp,
        bar_z=bar_z,
        x_child=x,
        z_child=z,
        g=loss_fn.g,
        delta=loss_fn.delta,
        phi=loss_fn.phi,
    )
    torch.testing.assert_close(residual, components["q_training_residual"])
    torch.testing.assert_close(
        components["q_training_residual"],
        components["q_target_total"] - Q,
    )
    torch.testing.assert_close(
        components["q_issue_minus_target"],
        -components["q_training_residual"],
    )
    torch.testing.assert_close(
        components["q_pricing_residual"],
        components["q_training_residual"],
    )
    torch.testing.assert_close(
        components["q_target_total"],
        components["q_target_survival"] + components["q_target_recovery"],
    )


def test_q_residual_signs_cover_positive_and_negative_cases():
    components = compute_q_survival_recovery_components(
        Q=torch.tensor([[0.0], [0.5]], dtype=torch.float64),
        b=torch.tensor([[1.0], [0.0]], dtype=torch.float64),
        bar_i=torch.zeros(2, 1, dtype=torch.float64),
        M=torch.ones(2, 1, dtype=torch.float64),
        Qsp=torch.zeros(2, 1, dtype=torch.float64),
        bar_z=torch.zeros(2, 1, dtype=torch.float64),
        x_child=torch.zeros(2, 1, dtype=torch.float64),
        z_child=torch.zeros(2, 1, dtype=torch.float64),
        g=Config.G,
        delta=Config.DELTA,
        phi=Config.PHI,
    )

    torch.testing.assert_close(
        components["q_training_residual"],
        components["q_target_total"] - torch.tensor([[0.0], [0.5]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        components["q_issue_minus_target"],
        -components["q_training_residual"],
    )
    assert components["q_training_residual"][0].item() > 0.0
    assert components["q_training_residual"][1].item() < 0.0


def test_q_csv_validator_checks_finiteness_and_sign_identity():
    frame = torch.tensor(
        [
            [0.2, 0.5],
            [0.8, 0.3],
        ],
        dtype=torch.float64,
    )
    q_issue = frame[:, 0]
    q_target_total = frame[:, 1]
    q_training_residual = q_target_total - q_issue
    q_issue_minus_target = q_issue - q_target_total

    df = pd.DataFrame(
        {
            "q_issue": q_issue.tolist(),
            "q_target_total": q_target_total.tolist(),
            "q_training_residual": q_training_residual.tolist(),
            "q_issue_minus_target": q_issue_minus_target.tolist(),
            "bar_i": [0.2, 0.4],
            "bar_i_multiplier": [1.0 + 0.2 * (Config.G - 1.0), 1.0 + 0.4 * (Config.G - 1.0)],
            "bar_i_mode": ["fixed_parent", "fixed_parent"],
        }
    )
    validate_q_decomposition_frame(df)

    bad = df.copy()
    bad.loc[0, "q_issue_minus_target"] *= -1.0
    try:
        validate_q_decomposition_frame(bad)
    except RuntimeError as exc:
        assert "sign identity failed" in str(exc)
    else:
        raise AssertionError("validator should reject inconsistent residual signs")


class _BarIVariesWithDebtModel:
    def __call__(self, state):
        return SimpleNamespace(bar_i=(0.1 + 0.7 * state[:, 0:1]).clamp(0.0, 1.0))


def test_bar_i_mode_fixed_parent_vs_recompute_issue_state():
    model = _BarIVariesWithDebtModel()
    parent = torch.tensor([[0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    issue_states = parent.expand(3, -1).clone()
    issue_states[:, 0:1] = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)

    fixed = compute_issue_bar_i(
        model,
        parent,
        issue_states,
        mode="fixed_parent",
    )
    recomputed = compute_issue_bar_i(
        model,
        parent,
        issue_states,
        mode="recompute_issue_state",
    )

    expected_fixed = torch.full_like(fixed, 0.1 + 0.7 * 0.2)
    expected_recomputed = 0.1 + 0.7 * issue_states[:, 0:1]
    torch.testing.assert_close(fixed, expected_fixed)
    torch.testing.assert_close(recomputed, expected_recomputed)
    assert torch.unique(fixed).numel() == 1
    assert torch.unique(recomputed).numel() == issue_states.shape[0]


def test_default_boundary_censoring_and_monotonicity_use_observed_only():
    rows = []
    # Observed boundary at z=0.0.
    for z, survival in [(-1.0, 0.0), (0.0, 0.7), (1.0, 1.0)]:
        rows.append({"branch": "p0", "b_candidate": 0.0, "z_child": z, "survival_child": survival})
    # Entire grid survives: boundary lies below z_min.
    for z in [-1.0, 0.0, 1.0]:
        rows.append({"branch": "p0", "b_candidate": 0.5, "z_child": z, "survival_child": 1.0})
    # Observed boundary at z=-1.0; this would be a violation if censored rows
    # were dropped before checking adjacency.
    for z, survival in [(-1.0, 0.8), (0.0, 1.0), (1.0, 1.0)]:
        rows.append({"branch": "p0", "b_candidate": 1.0, "z_child": z, "survival_child": survival})
    # Entire grid defaults: boundary lies above z_max.
    for z in [-1.0, 0.0, 1.0]:
        rows.append({"branch": "p0", "b_candidate": 1.5, "z_child": z, "survival_child": 0.0})

    boundary = make_default_boundary_frame(
        pd.DataFrame(rows),
        episode=2,
        source_index=7,
        z_min=-1.0,
        z_max=1.0,
    )

    by_b = boundary.set_index("b_candidate")
    assert by_b.loc[0.0, "boundary_censoring"] == "observed"
    assert by_b.loc[0.0, "boundary_found"]
    assert by_b.loc[0.0, "z_survival_boundary"] == 0.0
    assert by_b.loc[0.5, "boundary_censoring"] == "left_censored"
    assert pd.isna(by_b.loc[0.5, "z_survival_boundary"])
    assert by_b.loc[0.5, "z_boundary_upper_bound"] == -1.0
    assert by_b.loc[1.5, "boundary_censoring"] == "right_censored"
    assert pd.isna(by_b.loc[1.5, "z_survival_boundary"])
    assert by_b.loc[1.5, "z_boundary_lower_bound"] == 1.0
    assert int(boundary["boundary_monotonicity_violation_count"].iloc[0]) == 0
