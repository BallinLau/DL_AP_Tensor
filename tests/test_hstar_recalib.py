"""hstar_recalib_count / hstar_recalib_every 参数化回归。

- 默认 0：不触发周期校正，输出与不传新参数逐位一致（旧行为不变）；
- count=K：校正事件恰在换算点 every=max(1, horizon//(K+1)) 的
  t=every,2*every,... 发生（t=0 初始校正另计）；
- HyperParams -> _prepare_simulation_kwargs -> SimulateTS 透传正确，
  count/every 冲突时 count 优先并打 warning（换算规则仅 SimulateTS.__init__ 一处）。
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config.hyperparams import HyperParams  # noqa: E402
from data.entry import INVESTMENT_COMPARE_SPEC_VERSION  # noqa: E402
from data.simulate_ts import SimulateTS  # noqa: E402
from training.episode import Episode  # noqa: E402


class _AlwaysInvestPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward_value_components(self, state):
        v0 = torch.ones_like(state[:, 0:1]) + self.anchor * 0.0
        return {
            "V0_physical": v0,
            "VI_physical": v0 + 1.0,
            "V0_normalized": 100.0 * v0,
            "VI_normalized": -100.0 * (v0 + 1.0),
        }

    def forward_simulation(self, state):
        n = state.shape[0]
        zeros = state.new_zeros((n, 1)) + self.anchor * 0.0
        ones = torch.ones_like(zeros)
        bp = zeros + 0.4
        return SimpleNamespace(
            Q=zeros,
            P0=ones,
            PI=ones + 1.0,
            bar_i=ones,
            bar_z=zeros,
            P=ones,
            bp0=bp,
            bpI=bp,
            bp=bp,
        )


class _PositiveDeltaConfig:
    DEVICE = torch.device("cpu")
    RHO_X = 0.0
    SIGMA_X = 0.0
    XBAR = 0.0
    RHO_Z = 0.0
    SIGMA_Z = 0.0
    ZBAR = 0.0
    DELTA = 0.0
    PHI = 1.0
    G = 1.5
    I_THRESHOLD = 0.5
    ZETA = 0.25
    SIM_B_INIT_MIN = 0.0
    SIM_B_INIT_MAX = 1.0
    ENTRY_B_MIN = 0.0
    ENTRY_B_MAX = 1.0


def _sim(*, tmp_path, n_paths=1, group_size=2, horizon=1, initial_hstar=True,
         **recalib):
    return SimulateTS(
        models={"policy_value": _AlwaysInvestPolicy()},
        config=_PositiveDeltaConfig,
        n_paths=n_paths,
        group_size=group_size,
        horizon=horizon,
        branch_num=1,
        enable_entry=True,
        enable_exit=True,
        device=torch.device("cpu"),
        entry_mode="investment_compare",
        entry_spec_version=INVESTMENT_COMPARE_SPEC_VERSION,
        entry_capital_ratio=0.0,
        entry_size_ratio=1.0,
        entry_cost_max=_PositiveDeltaConfig.I_THRESHOLD,
        entry_rng_seed=17,
        consumption_aggregation_mode="raw",
        consumption_floor=1e-4,
        node_accounting_mode="economic_node_ledger",
        transition_rng_mode="stable_firm_identity",
        common_transition_seed=424242,
        initial_hstar_enabled=initial_hstar,
        initial_hstar_residual_tolerance=1e-5,
        initial_hstar_parity_tolerance=1e-5,
        initial_hstar_coarse_points=10,
        initial_hstar_refinement_points=20,
        initial_hstar_max_refinement_rounds=4,
        initial_hstar_support_lower=-8.0,
        initial_hstar_support_upper=2.0,
        initial_hstar_width_tolerance=1e-5,
        initial_hstar_audit_output_dir=str(tmp_path),
        initial_hstar_audit_episode_id=4,
        **recalib,
    )


def _assert_bitwise_identical(left: torch.Tensor, right: torch.Tensor) -> None:
    # torch.equal 视 NaN != NaN；此处要求逐位一致，仅 NaN 位置视为相等。
    assert left.shape == right.shape
    same = (left == right) | (torch.isnan(left) & torch.isnan(right))
    assert bool(same.all()), (
        f"{int((~same).sum())} bitwise-differing elements of {left.numel()}"
    )


def test_default_zero_keeps_legacy_behavior_bitwise(tmp_path):
    torch.manual_seed(123)
    baseline = _sim(tmp_path=tmp_path, horizon=3).simulate_tensor()
    torch.manual_seed(123)
    explicit = _sim(
        tmp_path=tmp_path,
        horizon=3,
        hstar_recalib_count=0,
        hstar_recalib_every=0,
    ).simulate_tensor()
    for output in (baseline, explicit):
        assert output.meta["hstar_recalib_every"] == 0
        assert output.meta["hstar_recalib_events"] == []
        assert output.meta["initial_hstar"]["success_count"] == 1
    _assert_bitwise_identical(baseline.firm.data, explicit.firm.data)
    _assert_bitwise_identical(baseline.macro.data, explicit.macro.data)


def test_count_resolves_to_expected_event_times(tmp_path):
    torch.manual_seed(123)
    sim = _sim(tmp_path=tmp_path, horizon=6, hstar_recalib_count=2)
    # 换算规则：every = max(1, horizon // (count + 1)) = max(1, 6 // 3)。
    assert sim.hstar_recalib_count == 2
    assert sim.hstar_recalib_every == 2
    output = sim.simulate_tensor()
    events = output.meta["hstar_recalib_events"]
    assert output.meta["hstar_recalib_every"] == 2
    assert [event["t"] for event in events] == [2, 4]
    assert [event["scope"] for event in events] == [
        "t2_periodic_recalib",
        "t4_periodic_recalib",
    ]
    for event in events:
        assert event["status"] == "OK"
        assert event["polish_fast_path_enabled"] is True
        assert event["n_paths_healthy"] == 1
    assert any(int(event["n_paths_hstar_written"]) >= 1 for event in events)


@pytest.mark.parametrize(
    "horizon,count,every_in,expected",
    [
        (50, 4, 0, 10),
        (50, 9, 0, 5),
        (50, 49, 0, 1),
        (50, 200, 0, 1),
        (50, 0, 7, 7),
        (50, 0, 0, 0),
        (6, 2, 0, 2),
    ],
)
def test_count_to_every_conversion_table(tmp_path, horizon, count, every_in,
                                         expected):
    sim = _sim(
        tmp_path=tmp_path,
        horizon=horizon,
        hstar_recalib_count=count,
        hstar_recalib_every=every_in,
    )
    assert sim.hstar_recalib_every == expected


def test_count_wins_over_every_with_warning(tmp_path):
    with pytest.warns(UserWarning, match="count wins"):
        sim = _sim(
            tmp_path=tmp_path,
            horizon=50,
            hstar_recalib_count=4,
            hstar_recalib_every=3,
        )
    assert sim.hstar_recalib_count == 4
    assert sim.hstar_recalib_every == 10


def test_negative_values_rejected(tmp_path):
    with pytest.raises(ValueError, match="hstar_recalib_count"):
        _sim(tmp_path=tmp_path, hstar_recalib_count=-1)
    with pytest.raises(ValueError, match="hstar_recalib_every"):
        _sim(tmp_path=tmp_path, hstar_recalib_every=-1)


def test_hyperparams_to_simulate_ts_passthrough(tmp_path):
    episode = Episode.__new__(Episode)
    episode.episode_id = 1

    episode.hyperparams = HyperParams()
    kwargs = episode._prepare_simulation_kwargs({})
    assert kwargs["hstar_recalib_count"] == 0
    assert kwargs["hstar_recalib_every"] == 0

    episode.hyperparams = HyperParams(
        hstar_recalib_count=4,
        hstar_recalib_every=3,
    )
    kwargs = episode._prepare_simulation_kwargs({})
    assert kwargs["hstar_recalib_count"] == 4
    assert kwargs["hstar_recalib_every"] == 3
    # count/every 冲突时 count 优先（换算唯一处：SimulateTS.__init__）。
    with pytest.warns(UserWarning, match="count wins"):
        sim = _sim(
            tmp_path=tmp_path,
            horizon=50,
            hstar_recalib_count=kwargs["hstar_recalib_count"],
            hstar_recalib_every=kwargs["hstar_recalib_every"],
        )
    assert sim.hstar_recalib_every == 10
