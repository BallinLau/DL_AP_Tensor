from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.market_clearing import (  # noqa: E402
    MarketClearingConfig,
    MarketClearingController,
    MarketClearingError,
    empty_entry_candidates,
)


class _SyntheticController(MarketClearingController):
    """Exercise root classification without invoking an economic model."""

    def __init__(self, scenario: str):
        super().__init__(
            MarketClearingConfig(
                coarse_points=10,
                refinement_points=20,
                max_refinement_rounds=3,
                support_lower=-1.0,
                support_upper=1.0,
                residual_tolerance=1.0e-5,
                width_tolerance=1.0e-8,
            )
        )
        self.scenario = scenario

    @torch.inference_mode()
    def evaluate(self, sim, state, candidates, h, *, detail=False):
        del sim, candidates
        if self.scenario == "jump":
            g = torch.where(h < 0.0, h.new_full((), -7.0e-3), h.new_full((), 3.0e-3))
            policy = h >= 0.0
        elif self.scenario == "continuous":
            g = h - 0.137
            policy = torch.zeros_like(h, dtype=torch.bool)
        elif self.scenario == "none":
            g = h.square() + 0.25
            policy = torch.zeros_like(h, dtype=torch.bool)
        elif self.scenario == "root_and_jump":
            continuous = h + 0.5
            g = torch.where(h < 0.3, continuous, h.new_full((), -0.1))
            policy = h >= 0.3
        elif self.scenario == "nan_candidate":
            g = torch.where(h < 0.0, h.new_full((), -1.0), h.new_full((), 1.0))
            g = torch.where(h.abs() < 1.0e-7, h.new_full((), float("nan")), g)
            policy = h >= 0.0
        else:  # pragma: no cover
            raise AssertionError(self.scenario)

        capital = torch.full_like(h, 10.0)
        consumption = torch.exp(h) * capital - g
        finite = torch.isfinite(g) & torch.isfinite(consumption)
        feasible = finite & (consumption > 0.0)
        scale = torch.maximum(
            torch.ones_like(g), torch.maximum(capital.abs(), consumption.abs())
        )
        result = {
            "h": h,
            "G": g,
            "G_normalized": g / scale,
            "C_raw": consumption,
            "K_total": capital,
            "resource_feasible": feasible,
            "finite": finite,
            "investment_count": policy.to(torch.long),
            "default_count": policy.to(torch.long),
            "entry_count": policy.to(torch.long),
            "firm_count": torch.ones_like(h, dtype=torch.long),
        }
        if detail:
            B, H = h.shape
            policy_firm = policy[:, :, None].to(h.dtype)
            result.update(
                {
                    "accepted_entry": torch.empty(
                        (B, H, 0), device=h.device, dtype=torch.bool
                    ),
                    "bar_i_firm": policy_firm,
                    "bar_z_firm": policy_firm,
                    "bp_firm": torch.zeros_like(policy_firm),
                    "P0_firm": torch.zeros_like(policy_firm),
                    "PI_firm": torch.zeros_like(policy_firm),
                }
            )
        return result


def _state() -> dict[str, torch.Tensor]:
    return {
        "b": torch.zeros((1, 1)),
        "alive": torch.ones((1, 1), dtype=torch.bool),
        "hatcf": torch.tensor([0.1]),
    }


def _run_failure(scenario: str) -> dict:
    controller = _SyntheticController(scenario)
    state = _state()
    with pytest.raises(MarketClearingError, match="no valid root"):
        controller.clear_state(
            object(), state, empty_entry_candidates(state), t=3, branch=1
        )
    assert len(controller.failure_records) == 1
    return controller.failure_records[0]


def test_terminal_jump_rejected_retains_discontinuity_classification():
    record = _run_failure("jump")
    assert record["terminal_status"] == "DISCONTINUITY_CROSSING"
    assert record["discontinuity_bracket_count"] >= 1
    assert record["unresolved_bracket_count"] >= 1
    assert "RESIDUAL_TOLERANCE_FAILED" in record["candidate_rejection_reason"]


def test_continuous_crossing_is_accepted_as_root():
    controller = _SyntheticController("continuous")
    state = _state()
    cleared = controller.clear_state(
        object(), state, empty_entry_candidates(state), t=0, branch=-1
    )
    assert bool(cleared["market_root_found"][0])
    assert abs(float(cleared["hatc_cleared"][0]) - 0.137) < 1.0e-4
    assert controller.root_records[0]["terminal_status"] == "ROOT_FOUND"


def test_no_scanned_bracket_stays_no_bracket():
    record = _run_failure("none")
    assert record["terminal_status"] == "NO_BRACKET_ON_TESTED_GRID"
    assert record["bracket_count"] == 0
    assert record["discontinuity_bracket_count"] == 0


def test_same_parent_retains_valid_root_and_rejected_jump_evidence():
    controller = _SyntheticController("root_and_jump")
    state = _state()
    controller.clear_state(
        object(), state, empty_entry_candidates(state), t=0, branch=-1
    )
    record = controller.root_records[0]
    assert record["valid_root_count"] >= 1
    assert record["discontinuity_bracket_count"] >= 1
    assert record["unresolved_bracket_count"] >= 1
    assert "RESIDUAL_TOLERANCE_FAILED" in record["candidate_rejection_reason"]


def test_nonfinite_candidate_is_not_relabelled_as_discontinuity():
    record = _run_failure("nan_candidate")
    assert record["terminal_status"] != "DISCONTINUITY_CROSSING"
    assert "NONFINITE_CANDIDATE" in record["candidate_rejection_reason"]

