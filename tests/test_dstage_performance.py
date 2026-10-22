"""Optimizer-free inference checks; not a trained-checkpoint GPU certificate."""
from dataclasses import replace
import pytest
import torch
from data.market_clearing import MarketClearingConfig, MarketClearingController
from models.policy_value import PolicyValueModel


@pytest.mark.parametrize('q_mode', ['direct', 'b_times_unit', 'hybrid_regime'])
@pytest.mark.parametrize('scale', ['none', 'exp_xz'])
def test_lightweight_values_bitwise_equal(q_mode, scale):
    torch.manual_seed(7)
    model = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8,
                             q_parameterization=q_mode, value_scale_mode=scale).eval()
    states = torch.randn(19, 7)
    states[:, 0] = torch.linspace(0, 1, 19)
    states[:, 2] = torch.arange(19) % 2
    states[:, 3] = torch.linspace(0, 1, 19)
    states[:, 4] = -2
    states[:, 5] = torch.linspace(-8, 2, 19)
    controller = MarketClearingController(MarketClearingConfig(candidate_chunk_size=7))
    original = controller._firm_policy(model, states)
    optimized = controller._hard_candidate_values(model, states)
    for key in ['P0', 'PI', 'Phat']:
        assert torch.equal(original[key], optimized[key]), key
    assert torch.equal(original['Phat'] > 0, optimized['Phat'] > 0)
    assert torch.equal(original['PI'] >= original['P0'], optimized['PI'] >= optimized['P0'])


def test_unknown_model_keeps_full_forward():
    class Custom(PolicyValueModel):
        pass
    model = Custom(share_hidden_dims=[8], share_output_dim=8).eval()
    controller = MarketClearingController(MarketClearingConfig())
    values = controller._hard_candidate_values(model, torch.zeros(2, 7))
    assert 'bp' in values and 'bar_i' in values


def test_fast_path_requires_explicit_configuration():
    config = MarketClearingConfig()
    assert not config.lightweight_hard_candidates
    assert not config.profile_root_timing
    assert replace(config, lightweight_hard_candidates=True).residual_tolerance == 1e-5


def test_full_candidate_ledger_and_detail_route():
    from test_entry_value_cost import _value_cost_sim
    from data.simulate_ts_parallel import _initialize_batched_state
    from data.market_clearing import empty_entry_candidates
    sim, _ = _value_cost_sim()
    sim.models['policy_value'] = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).eval()
    sim.hard_realized_aggregation = True
    state = _initialize_batched_state(sim, 2)
    entries = empty_entry_candidates(state)
    h = torch.tensor([[-8., -1., 2.]]).expand(len(state['b']), -1)
    original = MarketClearingController(MarketClearingConfig())
    optimized = MarketClearingController(MarketClearingConfig(lightweight_hard_candidates=True))
    a = original.evaluate(sim,state,entries,h)
    b = optimized.evaluate(sim,state,entries,h)
    for key in a:
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0,equal_nan=True)
    # Final detail must still contain actual full BP/head and policy diagnostics.
    a = original.evaluate(sim,state,entries,h,detail=True)
    b = optimized.evaluate(sim,state,entries,h,detail=True)
    for key in a:
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0,equal_nan=True)
