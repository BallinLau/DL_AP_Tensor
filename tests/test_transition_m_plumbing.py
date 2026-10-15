from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import inspect
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from config.hyperparams import HyperParams  # noqa: E402
from models.policy_value import PolicyValueModel  # noqa: E402
from training.episode import Episode  # noqa: E402
from training.inner_fixed_point import (  # noqa: E402
    _transition_m_summary,
    evaluate_inner_operator_snapshot,
)


def _m_episode(**overrides) -> Episode:
    episode = object.__new__(Episode)
    values = {
        "pv_fixed_sdf": False,
        "pv_fixed_sdf_value": 0.98,
        "pv_use_clipped_m": True,
        "pv_m_clamp_min": 0.7,
        "pv_m_clamp_max": 1.3,
        "q_use_detached_m": True,
        "q_m_clamp_min": 0.5,
        "q_m_clamp_max": 1.5,
        "pv_exact_eta_integration_enabled": True,
    }
    values.update(overrides)
    episode.hyperparams = SimpleNamespace(**values)
    episode.config = Config
    episode.device = torch.device("cpu")
    return episode


def _children() -> list[torch.Tensor]:
    state = torch.zeros((3, 7), dtype=torch.float32)
    child0 = torch.cat([state.clone(), torch.tensor([[0.81], [0.92], [1.03]])], dim=1)
    child1 = torch.cat([state.clone(), torch.tensor([[0.88], [0.97], [1.11]])], dim=1)
    return [child0, child1]


def _full_episode() -> Episode:
    device = torch.device("cpu")
    torch.manual_seed(20261015)
    hp = HyperParams()
    hp.pv_bp_training_mode = "target_grid"
    hp.pv_training_flow = "staged"
    hp.bp_grid_coarse_size = 3
    hp.bp_grid_refine_enabled = False
    hp.bp_grid_candidate_chunk_size = 2
    hp.bp_grid_parent_chunk_size = 2
    hp.bp_grid_max_expanded_states = 32
    hp.pv_use_clipped_m = False
    hp.pv_exact_eta_integration_enabled = True
    online = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
    target = deepcopy(online).eval()
    optimizer = torch.optim.AdamW(online.parameters(), lr=1e-3)
    return Episode(
        models={"policy_value": online},
        optimizers={"policy_value": optimizer},
        config=Config,
        hyperparams=hp,
        device=device,
        firm_target=target,
    )


def _target_batch(parent_width: int, m_shift: float = 0.0) -> dict:
    state = torch.tensor(
        [
            [0.15, -0.10, 1.0, 0.25, 0.10, -2.0, 4.0],
            [0.35, 0.20, 0.0, 0.40, -0.05, -1.8, 4.2],
        ],
        dtype=torch.float32,
    )
    parent = state if parent_width == 7 else torch.cat(
        [state, torch.full((2, 1), 999.0)], dim=1
    )
    child0 = state.clone()
    child1 = state.clone()
    child0[:, 1] += 0.07
    child1[:, 1] -= 0.05
    m0 = torch.tensor([[0.82], [0.91]]) + m_shift
    m1 = torch.tensor([[1.08], [1.17]]) + m_shift
    return {
        "parent": parent,
        "children": [torch.cat([child0, m0], dim=1), torch.cat([child1, m1], dim=1)],
    }


def test_parent7_child8_extracts_actual_transition_m():
    episode = _m_episode(pv_use_clipped_m=False)
    parent = torch.zeros((3, 7))
    raw, used, source = episode._build_policy_m_lists(
        parent, _children(), 0.7, 1.3,
        allow_unit_fallback=False, return_source=True,
    )
    assert source == "child_column_7"
    for child, raw_m, used_m in zip(_children(), raw, used):
        torch.testing.assert_close(raw_m, child[:, 7:8], rtol=0, atol=0)
        torch.testing.assert_close(used_m, child[:, 7:8], rtol=0, atol=0)


@pytest.mark.parametrize("extra", [0.0, 1.0, 999.0, -123.0])
def test_parent_layout_and_extra_column_never_change_transition_m(extra):
    episode = _m_episode(pv_use_clipped_m=False)
    children = _children()
    parent7 = torch.zeros((3, 7))
    parent8 = torch.cat([parent7, torch.full((3, 1), extra)], dim=1)
    raw7, used7 = episode._build_policy_m_lists(parent7, children, 0.7, 1.3)
    raw8, used8 = episode._build_policy_m_lists(parent8, children, 0.7, 1.3)
    for lhs, rhs in zip(raw7 + used7, raw8 + used8):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)


def test_legacy_child7_fallback_is_explicit_and_mixed_width_fails():
    episode = _m_episode()
    child7 = torch.zeros((3, 7))
    raw, source = episode._extract_transition_m_list(
        [child7, child7.clone()], 3, allow_unit_fallback=True
    )
    assert source == "unit_fallback"
    assert all(torch.equal(item, torch.ones((3, 1))) for item in raw)
    with pytest.raises(RuntimeError, match="Transition M is required"):
        episode._extract_transition_m_list([child7], 3, allow_unit_fallback=False)
    with pytest.raises(ValueError, match="Mixed child transition-M layouts"):
        episode._extract_transition_m_list(
            [_children()[0], child7], 3, allow_unit_fallback=True
        )


def test_p_clipping_and_fixed_sdf_keep_raw_transition_m():
    raw_values = torch.tensor([[0.5], [0.8], [1.4]])
    child = torch.cat([torch.zeros((3, 7)), raw_values], dim=1)
    episode = _m_episode()
    raw, used = episode._build_policy_m_lists(
        torch.zeros((3, 7)), [child], 0.7, 1.3, allow_unit_fallback=False
    )
    torch.testing.assert_close(raw[0], raw_values)
    torch.testing.assert_close(used[0], torch.tensor([[0.7], [0.8], [1.3]]))

    fixed = _m_episode(pv_fixed_sdf=True, pv_fixed_sdf_value=0.96)
    raw_fixed, used_fixed = fixed._build_policy_m_lists(
        torch.zeros((3, 7)), [child], 0.7, 1.3, allow_unit_fallback=False
    )
    torch.testing.assert_close(raw_fixed[0], raw_values)
    torch.testing.assert_close(used_fixed[0], torch.full_like(raw_values, 0.96))


def test_q_builder_clips_identically_for_training_and_evaluator_callers():
    values = torch.tensor([[0.3], [0.8], [1.8]])
    child = torch.cat([torch.zeros((3, 7)), values], dim=1)
    episode = _m_episode(q_m_clamp_min=0.5, q_m_clamp_max=1.5)
    train_raw, train_used, train_source = episode._build_q_m_lists(
        [child], 3, allow_unit_fallback=False, return_source=True
    )
    eval_raw, eval_used, eval_source = episode._build_q_m_lists(
        [child], 3, allow_unit_fallback=False, return_source=True
    )
    assert train_source == eval_source == "child_column_7"
    torch.testing.assert_close(train_raw[0], values)
    torch.testing.assert_close(train_used[0], torch.tensor([[0.5], [0.8], [1.5]]))
    torch.testing.assert_close(train_used[0], eval_used[0], rtol=0, atol=0)


def test_exact_eta_expansion_inherits_identical_transition_m_per_pair():
    episode = _m_episode()
    children = _children()
    raw, used = episode._build_policy_m_lists(
        torch.zeros((3, 7)), children, 0.7, 1.3, allow_unit_fallback=False
    )
    expanded_children, expanded_raw, expanded_used, weights = (
        episode._expand_policy_expectation_children(children, raw, used)
    )
    assert len(expanded_children) == 4
    assert weights is not None
    torch.testing.assert_close(expanded_raw[0], expanded_raw[1], rtol=0, atol=0)
    torch.testing.assert_close(expanded_raw[2], expanded_raw[3], rtol=0, atol=0)
    torch.testing.assert_close(expanded_used[0], expanded_used[1], rtol=0, atol=0)
    torch.testing.assert_close(expanded_used[2], expanded_used[3], rtol=0, atol=0)


def test_outer_inner_p_target_parity_and_cache_hash_tracks_transition_m():
    episode = _full_episode()
    outer = _target_batch(8)
    inner = _target_batch(7)
    outer_cache, outer_summary = episode._build_pq_value_target_cache(
        [outer], episode.firm_target
    )
    inner_cache, inner_summary = episode._build_pq_value_target_cache(
        [inner], episode.firm_target
    )
    assert outer_summary["p_m_source"] == inner_summary["p_m_source"] == "child_column_7"
    torch.testing.assert_close(
        outer_cache[0].p0_value_target, inner_cache[0].p0_value_target,
        rtol=0, atol=1e-6,
    )
    torch.testing.assert_close(
        outer_cache[0].pi_value_target, inner_cache[0].pi_value_target,
        rtol=0, atol=1e-6,
    )

    changed_cache, _ = episode._build_pq_value_target_cache(
        [_target_batch(7, m_shift=-0.20)], episode.firm_target
    )
    assert inner_cache[0].m_hash != changed_cache[0].m_hash
    assert not torch.allclose(
        inner_cache[0].p0_value_target, changed_cache[0].p0_value_target
    )
    assert not torch.allclose(
        inner_cache[0].pi_value_target, changed_cache[0].pi_value_target
    )


def test_q_economic_target_is_m_sensitive_and_train_eval_semantics_match():
    episode = _full_episode()
    q_loss = episode.loss_fns["q"]
    child = torch.cat([torch.zeros((2, 7)), torch.tensor([[0.8], [1.2]])], dim=1)
    _, train_m, _ = episode._build_q_m_lists(
        [child], 2, allow_unit_fallback=False, return_source=True
    )
    _, eval_m, _ = episode._build_q_m_lists(
        [child], 2, allow_unit_fallback=False, return_source=True
    )
    common = dict(
        Q=torch.tensor([[0.2], [0.3]]),
        b=torch.tensor([[0.4], [0.5]]),
        bar_i=torch.tensor([[0.2], [0.3]]),
        Qsp_children=[torch.tensor([[0.1], [0.15]])],
        bar_z_children=[torch.tensor([[0.1], [0.2]])],
        x_children=[torch.tensor([[0.0], [0.1]])],
        z_children=[torch.tensor([[0.0], [0.1]])],
    )
    train = q_loss.compute_main_components(M_list=train_m, **common)[0]
    evaluated = q_loss.compute_main_components(M_list=eval_m, **common)[0]
    for key in ("q_target_total", "continuation_q_component", "recursion_gain"):
        torch.testing.assert_close(train[key], evaluated[key], rtol=0, atol=0)

    unit = q_loss.compute_main_components(
        M_list=[torch.ones_like(train_m[0])], **common
    )[0]
    for key in ("q_target_total", "continuation_q_component", "recursion_gain"):
        assert not torch.allclose(train[key], unit[key])


def test_formal_inner_evaluator_uses_shared_p_and_q_builders_and_rejects_fallback():
    source = inspect.getsource(evaluate_inner_operator_snapshot)
    assert "_build_policy_m_lists" in source
    assert "_build_q_m_lists" in source
    assert "allow_unit_fallback=False" in source
    assert "child[:, 7:8]" not in source
    with pytest.raises(RuntimeError, match="forbidden unit-M fallback"):
        _transition_m_summary(
            "p", ["unit_fallback"], [torch.ones(1)], [torch.ones(1)]
        )
