"""生产算子 CPU 集成验证：STRICT_GLOBAL t0 h-star 与真实经济账本。

复用 test_entry_investment_compare 的微型 CPU 模拟配置（investment_compare
entry、raw 聚合、economic_node_ledger），端到端验证：

- raw 出清根在 declared support 内被找到，且候选/正式 t0 生产账本、
  Firm 决策与资源账本 parity 通过；
- floor 语义：floor flag 记录、raw 出清不受 floor 影响；
- 无根时逐 Path 审计持久化且不伪造 Parent；
- 硬破产（bar_z）随 hatcf 分段跳变时的多根/跳变处理；
- 固定 CRN：校准不改变生产转移随机冲击。
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.entry import INVESTMENT_COMPARE_SPEC_VERSION  # noqa: E402
from data.simulate_ts import SimulateTS  # noqa: E402


class _BranchValuePolicy(nn.Module):
    def __init__(self, *, i_threshold: float = 0.5):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.i_threshold = float(i_threshold)

    def forward_value_components(self, state):
        v0 = torch.ones_like(state[:, 0:1]) + self.anchor * 0.0
        return {
            "V0_physical": v0,
            "VI_physical": v0 + 1.0,
            "V0_normalized": 100.0 * v0,
            "VI_normalized": -100.0 * (v0 + 1.0),
        }


class _AlwaysInvestPolicy(_BranchValuePolicy):
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


class _HardDefaultSwitchPolicy(_BranchValuePolicy):
    """hatcf>0 时硬破产（bar_z=1、bar_i=0），hatcf<=0 时全额投资。

    C_raw 与 K 账本在 h=0 处跳变，构成硬决策分段经济。
    """

    def forward_simulation(self, state):
        n = state.shape[0]
        tail = (state[:, 5:6] > 0.0).to(state.dtype)
        zeros = state.new_zeros((n, 1)) + self.anchor * 0.0
        ones = torch.ones_like(zeros)
        bp = zeros + 0.4
        return SimpleNamespace(
            Q=zeros,
            P0=ones,
            PI=ones + 1.0,
            bar_i=1.0 - tail,
            bar_z=tail,
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


class _HugeZConfig(_PositiveDeltaConfig):
    ZBAR = 10.0


def _sim(*, policy, config, n_paths=1, group_size=2, horizon=1, tmp_path,
         initial_hstar=True, floor=1e-4, seed=123):
    return SimulateTS(
        models={"policy_value": policy},
        config=config,
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
        entry_cost_max=config.I_THRESHOLD,
        entry_rng_seed=17,
        consumption_aggregation_mode="raw",
        consumption_floor=floor,
        node_accounting_mode="economic_node_ledger",
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
    )


def test_root_found_with_candidate_production_parity(tmp_path):
    torch.manual_seed(123)
    simulator = _sim(
        policy=_AlwaysInvestPolicy(),
        config=_PositiveDeltaConfig,
        tmp_path=tmp_path,
    )
    output = simulator.simulate_tensor()
    diagnostics = output.meta["initial_hstar"]
    assert diagnostics["success_count"] == 1
    assert diagnostics["production_raw_clearing_passed"] is True
    assert diagnostics["production_ledger_parity_passed"] is True
    assert diagnostics["production_firm_decision_parity_passed"] is True
    audit_file = Path(tmp_path) / "initial_hstar_audit_4.json"
    assert audit_file.is_file()
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    record = audit["results"][0]
    assert record["terminal_status"] == "ROOT_FOUND"
    # raw G 验收：exp(h)*K - C_raw 残差在容差内（audit 记录）。
    assert abs(record["normalized_residual"]) <= 1e-5


def test_floor_flag_recorded_and_raw_clearing_unaffected(tmp_path):
    torch.manual_seed(123)
    simulator = _sim(
        policy=_AlwaysInvestPolicy(),
        config=_PositiveDeltaConfig,
        tmp_path=tmp_path,
        floor=1e6,
    )
    output = simulator.simulate_tensor()
    diagnostics = output.meta["initial_hstar"]
    assert diagnostics["success_count"] == 1
    audit = json.loads(
        (Path(tmp_path) / "initial_hstar_audit_4.json").read_text(encoding="utf-8")
    )
    record = audit["results"][0]
    assert record["terminal_status"] == "ROOT_FOUND"
    # 大 floor 使 exported hatc_cal 使用 clamp 后的 C，但 raw 出清与审计仍成立。
    assert diagnostics["floor_applied_root_count"] == 1


def test_no_root_raises_and_persists_audit(tmp_path):
    torch.manual_seed(123)
    simulator = _sim(
        policy=_AlwaysInvestPolicy(),
        config=_HugeZConfig,
        tmp_path=tmp_path,
    )
    with pytest.raises(RuntimeError, match="no valid clearing root"):
        simulator.simulate()
    audit = json.loads(
        (Path(tmp_path) / "initial_hstar_audit_4.json").read_text(encoding="utf-8")
    )
    record = audit["results"][0]
    assert record["terminal_status"].startswith("NO_ROOT_")
    assert record["failure_reason"]


def test_hard_default_switch_finds_root_and_records_jump(tmp_path):
    torch.manual_seed(123)
    simulator = _sim(
        policy=_HardDefaultSwitchPolicy(),
        config=_PositiveDeltaConfig,
        tmp_path=tmp_path,
    )
    output = simulator.simulate_tensor()
    diagnostics = output.meta["initial_hstar"]
    assert diagnostics["success_count"] == 1
    audit = json.loads(
        (Path(tmp_path) / "initial_hstar_audit_4.json").read_text(encoding="utf-8")
    )
    record = audit["results"][0]
    assert record["terminal_status"] == "ROOT_FOUND"
    assert (
        record["discontinuity_bracket_count"] >= 1
        or record["unresolved_bracket_count"] >= 1
        or record["root_count"] >= 2
    )


def test_calibration_preserves_production_transition_crn(tmp_path):
    def run(with_calibration):
        torch.manual_seed(321)
        simulator = _sim(
            policy=_AlwaysInvestPolicy(),
            config=_PositiveDeltaConfig,
            n_paths=2,
            tmp_path=tmp_path,
            initial_hstar=with_calibration,
        )
        return simulator.simulate_tensor()

    calibrated = run(True)
    baseline = run(False)
    assert calibrated.meta["initial_hstar"]["success_count"] == 2

    def eps_column(output):
        columns = output.macro.columns
        index = columns.index("eps_x_transition")
        return output.macro.data[:, index]

    eps_cal = eps_column(calibrated)
    eps_base = eps_column(baseline)
    assert eps_cal.shape == eps_base.shape
    valid = torch.isfinite(eps_cal) & torch.isfinite(eps_base)
    assert valid.sum() > 0
    assert torch.equal(eps_cal[valid], eps_base[valid])