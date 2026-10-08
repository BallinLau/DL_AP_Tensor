from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from config import Config
from data import simulate_ts_parallel
from data.simulate_ts import SimulateTS
from experiments.bp_head_vs_grid_simulation import (
    capture_rng_state,
    compose_grid_action,
    distribution_by_t,
    one_dimensional_wasserstein,
    restore_rng_state,
    rng_state_equal,
    validate_paired_random_inputs,
)
from training.bp_simulation_policy import GridBPSimulationPolicy


class _DummyPolicy(torch.nn.Module):
    def forward(self, state):  # pragma: no cover - simulation helper is patched
        raise AssertionError("patched simulation forward should be used")


def _policy_output(state: torch.Tensor) -> SimpleNamespace:
    n = int(state.shape[0])
    zeros = torch.zeros(n, 1, device=state.device)
    return SimpleNamespace(
        Q=zeros,
        P0=zeros,
        PI=zeros,
        bar_i=zeros,
        bar_z=zeros,
        P=zeros,
        bp0=torch.full_like(zeros, 0.2),
        bpI=torch.full_like(zeros, 0.8),
        bp=torch.full_like(zeros, 0.6),
    )


def _parallel_state() -> dict[str, torch.Tensor]:
    return {
        "x": torch.zeros(1),
        "b": torch.tensor([[0.7, 0.4]]),
        "z": torch.zeros(1, 2),
        "eta": torch.tensor([[1.0, 0.0]]),
        "i": torch.zeros(1, 2),
        "K": torch.ones(1, 2),
        "hatcf": torch.zeros(1),
        "lnkf": torch.zeros(1),
        "M": torch.ones(1),
        "alive": torch.ones(1, 2, dtype=torch.bool),
        "entry": torch.zeros(1, 2),
        "firm_id": torch.arange(2).reshape(1, 2),
        "next_firm_id": torch.full((1,), 2),
        "bar_i": torch.zeros(1, 2),
        "bar_z": torch.zeros(1, 2),
        "bp": torch.tensor([[0.7, 0.4]]),
    }


def _sim(override=None, *, source="head", grid_policy=None) -> SimulateTS:
    return SimulateTS(
        models={"policy_value": _DummyPolicy(), "sdf_fc1": None, "fc2": None},
        config=Config,
        n_paths=1,
        group_size=2,
        horizon=1,
        branch_num=2,
        enable_entry=False,
        enable_exit=False,
        device=torch.device("cpu"),
        bp_action_override=override,
        bp_action_source=source,
        bp_grid_policy=grid_policy,
    )


def test_bp_action_override_changes_only_reported_and_transition_policy(monkeypatch):
    monkeypatch.setattr(
        simulate_ts_parallel,
        "forward_policy_value_for_simulation",
        lambda _model, state: _policy_output(state),
    )
    seen = {}

    def override(**context):
        seen.update(context)
        return torch.tensor([[0.3], [0.9]])

    sim = _sim(override)
    state = _parallel_state()
    rows, _ = simulate_ts_parallel._process_node_batched(sim, state, 0, -1)
    columns = {name: index for index, name in enumerate(sim.FIRM_COLUMNS)}

    torch.testing.assert_close(rows[:, columns["bp"]], torch.tensor([0.3, 0.9]))
    torch.testing.assert_close(rows[:, columns["bp0"]], torch.tensor([0.2, 0.2]))
    torch.testing.assert_close(rows[:, columns["bpI"]], torch.tensor([0.8, 0.8]))
    torch.testing.assert_close(state["bp"], torch.tensor([[0.3, 0.9]]))
    assert seen["t"] == 0
    assert seen["branch"] == -1

    branches = simulate_ts_parallel._expand_branches_batched(sim, state)
    # Current eta_t controls refinancing: row 0 takes the override, row 1 keeps b_t.
    expected = torch.tensor([[0.3, 0.4]])
    torch.testing.assert_close(branches[0]["b"], expected)
    torch.testing.assert_close(branches[1]["b"], expected)


def test_none_override_preserves_formal_head_output(monkeypatch):
    monkeypatch.setattr(
        simulate_ts_parallel,
        "forward_policy_value_for_simulation",
        lambda _model, state: _policy_output(state),
    )
    sim = _sim(None)
    state = _parallel_state()
    rows, _ = simulate_ts_parallel._process_node_batched(sim, state, 0, -1)
    bp_column = sim.FIRM_COLUMNS.index("bp")
    torch.testing.assert_close(rows[:, bp_column], torch.tensor([0.6, 0.6]))
    torch.testing.assert_close(state["bp"], torch.tensor([[0.6, 0.6]]))


def test_grid_source_uses_resolver_and_current_parent_eta(monkeypatch):
    monkeypatch.setattr(
        simulate_ts_parallel,
        "forward_policy_value_for_simulation",
        lambda _model, state: _policy_output(state),
    )
    calls = []

    def grid_policy(**context):
        calls.append((context["t"], context["branch"]))
        return torch.tensor([[0.3], [0.9]])

    sim = _sim(source="grid", grid_policy=grid_policy)
    state = _parallel_state()
    rows, _ = simulate_ts_parallel._process_node_batched(sim, state, 0, -1)
    bp_column = sim.FIRM_COLUMNS.index("bp")
    torch.testing.assert_close(rows[:, bp_column], torch.tensor([0.3, 0.9]))
    torch.testing.assert_close(state["bp"], torch.tensor([[0.3, 0.9]]))
    assert calls == [(0, -1)]

    branches = simulate_ts_parallel._expand_branches_batched(sim, state)
    expected = torch.tensor([[0.3, 0.4]])
    torch.testing.assert_close(branches[0]["b"], expected)
    torch.testing.assert_close(branches[1]["b"], expected)


class _GridTarget(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, state):
        n = state.shape[0]
        zero = torch.zeros(n, 1, device=state.device, dtype=state.dtype)
        return SimpleNamespace(
            bp0=zero + 0.2,
            bpI=zero + 0.8,
            bp=zero + 0.6,
            bp_cond=zero + 0.5,
            bar_i_cond=zero + 0.4,
            survival_prob=zero + 0.75,
        )


class _GridSDF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward_step(self, *, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical):
        del x_prev, return_physical
        shape = x_curr.shape
        one = torch.ones(shape, device=x_curr.device, dtype=x_curr.dtype)
        return one, one, one, torch.zeros_like(x_curr), torch.zeros_like(x_curr)


class _GridTeacher:
    coarse_size = 3
    fine_size = 2
    refine = True

    def __init__(self):
        self.devices = []

    def compute(self, parent_state, children, m_list, *, branch, **kwargs):
        del children, m_list, kwargs
        self.devices.append(parent_state.device)
        n = parent_state.shape[0]
        value = 0.3 if branch == "p0" else 0.7
        out = torch.full((n, 1), value, device=parent_state.device)
        return {
            "bp_star": out,
            "top2_margin": torch.ones_like(out),
            "regret": torch.zeros_like(out),
            "refi_active": parent_state[:, 2:3] > 0.5,
        }

    def forward_stats(self):
        return {"bp_parent_chunks": 1}


def test_production_grid_policy_is_deterministic_rng_neutral_and_immutable(monkeypatch):
    teacher = _GridTeacher()
    monkeypatch.setattr(
        "training.bp_simulation_policy.BPGridTeacher.from_hyperparams",
        lambda *_args, **_kwargs: teacher,
    )
    model = _GridTarget()
    sdf = _GridSDF()
    hp = SimpleNamespace(
        pv_use_clipped_m=False,
        bp_grid_coarse_size=3,
        bp_grid_fine_size=2,
    )
    rng_before = torch.random.get_rng_state().clone()
    policy = GridBPSimulationPolicy(
        target_model=model,
        sdf_fc1_model=sdf,
        p0_loss=object(),
        pi_loss=object(),
        hyperparams=hp,
        economic_config=Config,
        n_child_shocks=2,
        shock_seed=77,
        require_cuda=False,
    )
    firm_state = torch.tensor(
        [[0.2, -0.1, 1.0, 0.1, 0.0, -2.0, 4.0]], dtype=torch.float32
    )
    kwargs = {
        "firm_state": firm_state,
        "hatc_cal": torch.tensor([[-2.1]]),
        "lnk_cal": torch.tensor([[4.1]]),
    }
    first = policy.evaluate_actions(**kwargs)
    second = policy.evaluate_actions(**kwargs)

    torch.testing.assert_close(first["bp"], torch.tensor([[0.6]]))
    torch.testing.assert_close(second["bp"], first["bp"])
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert teacher.devices == [firm_state.device] * 4
    policy.verify_immutable()
    metrics = policy.instrumentation()
    assert metrics["calls"] == 2
    assert metrics["num_parent_states"] == 2
    assert metrics["coarse_eval_count"] == 12
    assert metrics["fine_eval_count"] == 8


def test_head_mode_is_backward_compatible_and_grid_rollout_is_deterministic(monkeypatch):
    monkeypatch.setattr(
        simulate_ts_parallel,
        "forward_policy_value_for_simulation",
        lambda _model, state: _policy_output(state),
    )

    torch.manual_seed(991)
    implicit_head = _sim().simulate_tensor()
    torch.manual_seed(991)
    explicit_head = _sim(source="head").simulate_tensor()
    torch.testing.assert_close(
        implicit_head.firm.data, explicit_head.firm.data, equal_nan=True
    )
    torch.testing.assert_close(
        implicit_head.macro.data, explicit_head.macro.data, equal_nan=True
    )

    grid_policy = lambda **context: torch.full_like(context["bp_head"], 0.35)
    torch.manual_seed(992)
    first = _sim(source="grid", grid_policy=grid_policy).simulate_tensor()
    torch.manual_seed(992)
    second = _sim(source="grid", grid_policy=grid_policy).simulate_tensor()
    torch.testing.assert_close(first.firm.data, second.firm.data, equal_nan=True)
    torch.testing.assert_close(first.macro.data, second.macro.data, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_production_grid_policy_keeps_transition_and_teacher_tensors_on_cuda(monkeypatch):
    teacher = _GridTeacher()
    monkeypatch.setattr(
        "training.bp_simulation_policy.BPGridTeacher.from_hyperparams",
        lambda *_args, **_kwargs: teacher,
    )
    device = torch.device("cuda:0")
    policy = GridBPSimulationPolicy(
        target_model=_GridTarget().to(device),
        sdf_fc1_model=_GridSDF().to(device),
        p0_loss=object(),
        pi_loss=object(),
        hyperparams=SimpleNamespace(pv_use_clipped_m=False),
        economic_config=Config,
        n_child_shocks=2,
        shock_seed=77,
        require_cuda=True,
    )
    parent = torch.tensor(
        [[0.2, -0.1, 1.0, 0.1, 0.0, -2.0, 4.0]],
        device=device,
    )
    result = policy.evaluate_actions(
        firm_state=parent,
        hatc_cal=torch.tensor([[-2.1]], device=device),
        lnk_cal=torch.tensor([[4.1]], device=device),
    )

    assert policy._base_shock_bank.eps_x.device.type == "cuda"
    assert result["bp"].device.type == "cuda"
    assert all(value.type == "cuda" for value in teacher.devices)


def test_override_requires_policy_value_output():
    sim = _sim(lambda **_context: torch.tensor([[0.3], [0.9]]))
    sim.models["policy_value"] = None
    with pytest.raises(RuntimeError, match="requires a policy_value model output"):
        simulate_ts_parallel._process_node_batched(sim, _parallel_state(), 0, -1)


def test_rng_snapshot_replays_exact_random_tape():
    torch.manual_seed(123)
    np.random.seed(123)
    snapshot = capture_rng_state()
    first = (torch.rand(5), np.random.rand(5))
    restore_rng_state(snapshot)
    second = (torch.rand(5), np.random.rand(5))
    torch.testing.assert_close(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    restore_rng_state(snapshot)
    assert rng_state_equal(snapshot, capture_rng_state())


def test_grid_action_matches_formal_survival_mixing():
    p0 = torch.tensor([[0.1], [0.2]])
    mix = torch.tensor([[0.8], [0.9]])
    survival = torch.tensor([[0.0], [0.75]])
    actual = compose_grid_action(p0, mix, survival)
    torch.testing.assert_close(actual, torch.tensor([[0.1], [0.725]]))


def test_paired_random_input_validation_detects_shift():
    base = pd.DataFrame(
        {
            "path": [0, 0],
            "t": [0, 1],
            "ID": [1, 1],
            "x": [0.1, 0.2],
            "z": [0.3, 0.4],
            "ETA": [1.0, 0.0],
            "i": [0.02, 0.03],
            "entry": [0.0, 0.0],
        }
    )
    result = validate_paired_random_inputs(base, base.copy())
    assert result["passed"] is True
    shifted = base.copy()
    shifted.loc[1, "z"] += 1e-3
    with pytest.raises(RuntimeError, match="random-input replay mismatch"):
        validate_paired_random_inputs(base, shifted)


def test_wasserstein_known_shift():
    assert one_dimensional_wasserstein([0.0, 1.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_distribution_default_rate_uses_pre_exit_main_child_rows():
    parent = pd.DataFrame(
        {
            "path": [0], "t": [1], "branch": [-1], "ID": [1],
            "b": [0.2], "z": [0.0], "P": [1.0], "Bar_z": [0.0],
            "entry": [0.0], "Bar_i": [0.0], "K": [1.0],
        }
    )
    pre_exit = pd.DataFrame(
        {
            "path": [0, 0], "t": [1, 1], "branch": [0, 0], "ID": [1, 2],
            "b": [0.2, 0.4], "z": [0.0, 0.0], "P": [1.0, 0.0],
            "Bar_z": [0.0, 1.0], "entry": [0.0, 0.0],
            "Bar_i": [0.0, 0.0], "K": [1.0, 1.0],
        }
    )
    full = pd.concat([parent, pre_exit], ignore_index=True)
    summary = distribution_by_t(
        parent,
        parent.copy(),
        head_full=full,
        grid_full=full.copy(),
    )
    assert summary.loc[0, "default_rate_head"] == pytest.approx(0.5)
    assert summary.loc[0, "pre_exit_firm_count_head"] == 2
    assert summary.loc[0, "firm_count_head"] == 1
