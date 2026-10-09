from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluation.fixed_target_p_inner_solver import (
    classify_diagnosis,
    cycle_start_index,
    evaluate_fixed_cache,
    pure_value_regression_loss,
    surface_complexity,
    tensor_payload_sha256,
)
from experiments import fixed_target_p_inner_solver as runner


class _TinyValueModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value_encoder = torch.nn.Linear(2, 2, bias=False)
        self.v0_head = torch.nn.Linear(2, 1, bias=False)
        self.vi_head = torch.nn.Linear(2, 1, bias=False)
        self.barz_model = torch.nn.Linear(2, 1, bias=False)
        self.bari_model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.value_encoder.weight.copy_(torch.eye(2))
            self.v0_head.weight.copy_(torch.tensor([[1.0, 0.0]]))
            self.vi_head.weight.copy_(torch.tensor([[0.0, 1.0]]))

    def _value_outputs(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.value_encoder(state[:, :2])
        return self.v0_head(hidden), self.vi_head(hidden)


class _TinyEpisode:
    def __init__(self, *, normalize: bool = True) -> None:
        self.models = {"policy_value": _TinyValueModel()}
        self.hyperparams = SimpleNamespace(
            pv_bellman_normalize_by_value_scale=normalize,
            bp_grid_value_huber_delta=1.0,
        )
        self.weight_scheduler = {"p0": 2.0, "pi": 3.0}

    def _pv_value_scale(self, _model: torch.nn.Module, state: torch.Tensor) -> torch.Tensor:
        return torch.full((state.shape[0], 1), 2.0, dtype=state.dtype)

    @staticmethod
    def _huber_element(predicted: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
        return torch.nn.functional.huber_loss(
            predicted, target, delta=delta, reduction="none"
        )

    def _compute_cached_value_loss(self, batch: dict, item: SimpleNamespace):
        total, terms = pure_value_regression_loss(self, batch, item)
        # Distinct nonzero auxiliary terms make aggregation observable.
        terms = {
            **terms,
            "p0_cached_penalty_z": 0.25,
            "pi_cached_penalty_z": 0.50,
            "pi_cached_penalty_b": 0.75,
            "total": float(total.item()) + 1.5,
        }
        return total + 1.5, terms


def _cache(p0: list[float], pi: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        p0_value_target=torch.tensor(p0, dtype=torch.float32).reshape(-1, 1),
        pi_value_target=torch.tensor(pi, dtype=torch.float32).reshape(-1, 1),
    )


def test_reported_cycle_uses_previous_checkpoint() -> None:
    assert cycle_start_index(1) == 0
    assert cycle_start_index(5) == 4
    assert cycle_start_index(10) == 9
    with pytest.raises(ValueError):
        cycle_start_index(0)


def test_tensor_payload_hash_detects_target_change() -> None:
    first = {"target": torch.tensor([1.0, 2.0]), "meta": {"cycle": 1}}
    same = {"meta": {"cycle": 1}, "target": torch.tensor([1.0, 2.0])}
    changed = {"target": torch.tensor([1.0, 3.0]), "meta": {"cycle": 1}}
    assert tensor_payload_sha256(first) == tensor_payload_sha256(same)
    assert tensor_payload_sha256(first) != tensor_payload_sha256(changed)


def test_pure_regression_keeps_huber_and_value_scale_without_auxiliary_terms() -> None:
    episode = _TinyEpisode(normalize=True)
    batch = {"parent": torch.tensor([[2.0, 4.0], [4.0, 8.0]])}
    item = _cache([0.0, 0.0], [0.0, 0.0])
    total, terms = pure_value_regression_loss(episode, batch, item)
    p0_expected = torch.nn.functional.huber_loss(
        torch.tensor([[1.0], [2.0]]), torch.zeros(2, 1), reduction="mean", delta=1.0
    )
    pi_expected = torch.nn.functional.huber_loss(
        torch.tensor([[2.0], [4.0]]), torch.zeros(2, 1), reduction="mean", delta=1.0
    )
    assert total.item() == pytest.approx((2.0 * p0_expected + 3.0 * pi_expected).item())
    assert terms["p0_cached_penalty_z"] == 0.0
    assert terms["pi_cached_penalty_z"] == 0.0
    assert terms["pi_cached_penalty_b"] == 0.0


def test_evaluate_fixed_cache_uses_global_observations_and_separate_spaces() -> None:
    episode = _TinyEpisode(normalize=True)
    batches = [
        {"parent": torch.tensor([[1.0, 2.0]])},
        {"parent": torch.tensor([[3.0, 4.0], [5.0, 8.0]])},
    ]
    cache = [_cache([0.0], [0.0]), _cache([0.0, 0.0], [0.0, 0.0])]
    result = evaluate_fixed_cache(episode, batches, cache)
    p0_error = np.array([1.0, 3.0, 5.0])
    pi_error = np.array([2.0, 4.0, 8.0])
    combined = np.concatenate([p0_error, pi_error])
    assert result.metrics["combined_physical_rms"] == pytest.approx(
        np.sqrt(np.mean(combined**2))
    )
    assert result.metrics["combined_normalized_rms"] == pytest.approx(
        np.sqrt(np.mean((combined / 2.0) ** 2))
    )
    # Objective aggregation is row weighted: (1 * first + 2 * second) / 3.
    assert result.objective["p0_cached_penalty_z"] == pytest.approx(0.25)
    assert result.objective["pi_cached_penalty_b"] == pytest.approx(0.75)


def test_surface_complexity_reports_jumps_and_curvature() -> None:
    surface = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    summary = surface_complexity(surface, jump_threshold=0.1)
    assert summary["dynamic_range"] == 1.0
    assert summary["total_variation"] > 0.0
    assert summary["local_max_gradient"] == 1.0
    assert summary["argmax_jump_share"] > 0.0


def _fit(value: float) -> dict[str, float]:
    return {
        "train_combined_normalized_rms": value,
        "validation_combined_normalized_rms": value,
    }


def _complexity(multiplier: float = 1.0) -> dict:
    one = {
        "curvature_b": multiplier,
        "curvature_z": multiplier,
        "total_variation": multiplier,
    }
    return {
        1: {"p0_value_target": dict(one), "pi_value_target": dict(one)},
        10: {
            "p0_value_target": {key: value * multiplier for key, value in one.items()},
            "pi_value_target": {key: value * multiplier for key, value in one.items()},
        },
    }


def test_diagnosis_prefers_undertrained_when_long_production_fit_is_controlled() -> None:
    production = {1: {100: _fit(0.2), 1000: _fit(0.04)}, 10: {100: _fit(0.3), 1000: _fit(0.03)}}
    pure = {1: {100: _fit(0.1), 1000: _fit(0.01)}, 10: {100: _fit(0.1), 1000: _fit(0.01)}}
    result = classify_diagnosis(
        production=production, pure=pure,
        memorization={"1_batch": _fit(0.001)}, complexity=_complexity(),
    )
    assert result["primary"] == "A"


def test_diagnosis_detects_production_objective_conflict() -> None:
    production = {1: {1000: _fit(0.3)}, 10: {1000: _fit(0.4)}}
    pure = {1: {1000: _fit(0.01)}, 10: {1000: _fit(0.02)}}
    result = classify_diagnosis(
        production=production, pure=pure,
        memorization={"1_batch": _fit(0.001)}, complexity=_complexity(),
    )
    assert result["primary"] == "B"


def test_runner_source_constructs_cycle_target_once_before_budget_loop() -> None:
    source = inspect.getsource(runner.main)
    cache_position = source.index("train_cache = _build_p_caches")
    objective_loop = source.index('for objective, result_map in (("production"')
    assert cache_position < objective_loop
    assert 'f"cycle_{start_index:02d}.pt"' in source
    assert "target_cache_refreshed_during_training\": False" in source
