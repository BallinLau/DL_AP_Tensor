"""STRICT_GLOBAL 求根数学层测试（e35e2b8 修正）。

用合成 G(h)=exp(h)*K-C_raw(h) 曲面替换生产算子的两次评价入口，验证：

- 全局 declared support 扫描（不再只是 belief±4 局部二分）；
- 多重根去重与最近 belief 选择（低 h 决胜）；
- 硬决策跳变（jump）分类与候选拒绝；
- 逐 Path 失败审计持久化（无根/不可行不伪造 Parent）；
- raw 出清与 floor flag 记录（G 使用 pre-floor 账本语义）。
"""

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import data.simulate_ts_parallel as stp  # noqa: E402


# 每条 path 的分段 C_raw(h)：regions 按上界升序 (bound, c, inv, def)，K=2.0。
# path0 的分段边界保证每个粗网格单元只含一个变号事件，根与决策跳变都能被
# coarse-to-fine 细化发现。
PATH_ECONOMICS = {
    # 双根（-1.6094 与 ln(1.5)=0.4055）+ h=-1.0 决策跳变；每个粗网格单元
    # 只含一个变号事件（root/jump/root 分别落在三个相邻单元）。
    0: {
        "regions": [
            (-1.0, 0.4, 1, 0),
            (0.0, 100.0, 2, 1),
            (float("inf"), 3.0, 3, 2),
        ],
        "floor": False,
    },
    # 支撑区间内无变号。
    1: {"regions": [(float("inf"), 100.0, 1, 0)], "floor": False},
    # 单根 + floor。
    2: {"regions": [(float("inf"), 0.8, 1, 0)], "floor": True},
    # 处处经济不可行。
    3: {"regions": [(float("inf"), -1.0, 1, 0)], "floor": False},
}


def _c_raw(path: int, h: float) -> float:
    return _region_at(path, h)[0]


def _region_at(path: int, h: float):
    """(c, inv, default) 分段经济：region 按上界升序，h<bound 命中该段。"""
    regions = PATH_ECONOMICS[path]["regions"]
    for bound, c, inv, default in regions:
        if h < bound:
            return c, inv, default
    last = regions[-1]
    return last[1], last[2], last[3]


def _record(h: float, path: int) -> dict:
    c, inv, default = _region_at(path, h)
    k = 2.0
    g = torch.exp(torch.tensor(h, dtype=torch.float64)) * k - c
    scale = max(1.0, abs(c), abs(k))
    finite = torch.isfinite(torch.tensor(h)) and torch.isfinite(torch.tensor(c))
    feasible = finite and c > 0.0 and k > 0.0
    floor_flag = bool(PATH_ECONOMICS[path]["floor"])
    return {
        "h": torch.tensor(h, dtype=torch.float32),
        "G": torch.tensor(float(g), dtype=torch.float32),
        "G_norm": torch.tensor(float(g) / scale, dtype=torch.float32),
        "C_raw": torch.tensor(c, dtype=torch.float32),
        "C_eff": torch.tensor(max(c, 1e-4), dtype=torch.float32),
        "K": torch.tensor(k, dtype=torch.float32),
        "floor_applied": torch.tensor(floor_flag, dtype=torch.bool),
        "finite": torch.tensor(finite, dtype=torch.bool),
        "feasible": torch.tensor(feasible, dtype=torch.bool),
        "accounting_valid": torch.tensor(True, dtype=torch.bool),
        "resource_feasible": torch.tensor(feasible, dtype=torch.bool),
        "resource_residual": torch.tensor(0.0, dtype=torch.float32),
        "n_firms": torch.tensor(1.0, dtype=torch.float32),
        "inv_count": torch.tensor(inv, dtype=torch.long),
        "def_count": torch.tensor(default, dtype=torch.long),
        "entry_accepted": torch.tensor(0.0, dtype=torch.float32),
    }


def _economic_path(sim, path: int) -> int:
    return int(getattr(sim, "path_economics_map", {}).get(path, path))


def _fake_rows(sim, base_state, rows, h_rows, *, rng_cpu=None, rng_cuda=None,
               return_state=False, **kwargs):
    records = []
    failures = {}
    del base_state, rng_cpu, rng_cuda
    for index in range(int(rows.numel())):
        path = int(rows[index])
        economic = _economic_path(sim, path)
        records.append(_record(float(h_rows[index]), economic))
    return records, failures, {} if return_state else None


def _fake_grid(sim, base_state, rows, h_grid, *, rng_cpu=None, rng_cuda=None):
    names = (
        "G", "G_norm", "C_raw", "C_eff", "K", "feasible", "finite",
        "floor_applied", "inv_count", "def_count", "entry_accepted", "n_firms",
    )
    n_rows, n_cols = int(h_grid.shape[0]), int(h_grid.shape[1])
    columns = {name: [] for name in names}
    failures_list = []
    for column in range(n_cols):
        packed = {
            name: torch.full(
                (n_rows,),
                (
                    0
                    if name in {"inv_count", "def_count"}
                    else (
                        False
                        if name in {"feasible", "finite", "floor_applied"}
                        else float("nan")
                    )
                ),
                dtype=(
                    torch.bool
                    if name in {"feasible", "finite", "floor_applied"}
                    else (
                        torch.long
                        if name in {"inv_count", "def_count"}
                        else torch.float32
                    )
                ),
            )
            for name in names
        }
        for row in range(n_rows):
            path = int(rows[row])
            economic = _economic_path(sim, path)
            record = _record(float(h_grid[row, column]), economic)
            for name in names:
                packed[name][row] = record[name]
        for name in names:
            columns[name].append(packed[name])
        failures_list.append({})
    return {name: torch.stack(columns[name], dim=1) for name in names}, failures_list


def _sim(tmp_path, n_paths, beliefs) -> "_Sim":
    return _Sim(tmp_path, n_paths, beliefs)


class _Sim(SimpleNamespace):
    def __init__(self, tmp_path, n_paths, beliefs, path_economics_map=None):
        super().__init__()
        self.device = torch.device("cpu")
        self.n_paths = n_paths
        self.bp_action_source = "head"
        self.initial_hstar_residual_tolerance = 1e-5
        self.initial_hstar_parity_tolerance = 1e-5
        self.initial_hstar_coarse_points = 10
        self.initial_hstar_refinement_points = 20
        self.initial_hstar_max_refinement_rounds = 4
        self.initial_hstar_support_lower = -8.0
        self.initial_hstar_support_upper = 2.0
        self.initial_hstar_width_tolerance = 1e-5
        self.initial_hstar_audit_output_dir = str(tmp_path)
        self.initial_hstar_audit_episode_id = 4
        self.beliefs = beliefs
        self.path_economics_map = path_economics_map or {}


def _state(sim) -> dict:
    torch.manual_seed(7)
    n = sim.n_paths
    return {
        "b": torch.rand(n, 1),
        "alive": torch.ones(n, 1, dtype=torch.bool),
        "K": torch.full((n, 1), 2.0),
        "hatcf": torch.tensor(sim.beliefs, dtype=torch.float32),
        "lnkf": torch.full((n,), float("nan")),
    }


@pytest.fixture()
def patch_evaluators(monkeypatch):
    monkeypatch.setattr(stp, "_initial_hstar_evaluate_rows", _fake_rows)
    monkeypatch.setattr(stp, "_initial_hstar_grid_evaluate", _fake_grid)



def test_strict_global_multi_root_selection(tmp_path, patch_evaluators, monkeypatch):
    import sys as _sys_mod
    monkeypatch.setitem(
        _sys_mod.modules[__name__].PATH_ECONOMICS,
        1,
        {"regions": [(float("inf"), 0.8, 1, 0)], "floor": False},
    )
    sim = _sim(tmp_path, 2, [-2.0, 0.0])
    state = _state(sim)
    diagnostics = stp._calibrate_initial_hstar_batched(sim, state)

    assert diagnostics["success_count"] == 2
    assert (
        diagnostics["root_solver"]
        == "strict_global_batched_production_node_operator"
    )
    # path0 两个有效根 + 一个决策跳变区间；选择离 belief(-2) 更近的低根。
    assert diagnostics["multi_root_path_count"] == 1
    assert diagnostics["discontinuity_path_count"] >= 1
    assert diagnostics["floor_applied_root_count"] == 0
    audit_file = Path(tmp_path) / "initial_hstar_audit_4.json"
    assert audit_file.is_file()
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    by_path = {record["path"]: record for record in audit["results"]}
    assert by_path[0]["terminal_status"] == "ROOT_FOUND"
    assert by_path[0]["root_count"] == 2
    roots = by_path[0]["all_valid_roots"]
    assert len(roots) == 2
    assert min(roots) == pytest.approx(-1.6094, abs=1e-4)
    assert max(roots) == pytest.approx(0.4055, abs=1e-4)
    assert by_path[0]["selected_root"] == pytest.approx(min(roots))
    assert by_path[1]["terminal_status"] == "ROOT_FOUND"
    # 正式状态写入所选根。
    assert float(state["hatcf"][0]) == pytest.approx(-1.6094, abs=1e-4)
    assert float(state["hatcf"][1]) == pytest.approx(-0.9163, abs=1e-4)
    assert state["lnkf"].isfinite().all()


def test_strict_global_floor_recorded_at_root(tmp_path, patch_evaluators):
    # 单条 path 映射到经济 2（单根 + consumption-floor）；raw 出清根必须
    # 正常找到，floor flag 只记录在审计里，不阻塞 root。
    sim = _Sim(tmp_path, 1, [0.0], path_economics_map={0: 2})
    state = _state(sim)
    diagnostics = stp._calibrate_initial_hstar_batched(sim, state)
    assert diagnostics["success_count"] == 1
    assert diagnostics["floor_applied_root_count"] == 1
    audit = json.loads(
        (Path(tmp_path) / "initial_hstar_audit_4.json").read_text(encoding="utf-8")
    )
    record = audit["results"][0]
    assert record["terminal_status"] == "ROOT_FOUND"
    assert float(state["hatcf"][0]) == pytest.approx(-0.9163, abs=1e-4)


def test_strict_global_no_root_persists_per_path_audit(tmp_path, patch_evaluators):
    sim = _sim(tmp_path, 4, [0.0, 0.0, 0.0, 0.0])
    state = _state(sim)
    with pytest.raises(RuntimeError, match="no valid clearing root"):
        stp._calibrate_initial_hstar_batched(sim, state)
    audit_file = Path(tmp_path) / "initial_hstar_audit_4.json"
    assert audit_file.is_file()
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    statuses = {
        record["path"]: record["terminal_status"] for record in audit["results"]
    }
    assert statuses[0] == "ROOT_FOUND"
    assert statuses[2] == "ROOT_FOUND"
    assert statuses[1] == "NO_ROOT_NO_SIGN_CHANGE_IN_SUPPORT"
    assert statuses[3] == "NO_ROOT_NO_SIGN_CHANGE_IN_SUPPORT"
    failed = [record for record in audit["results"] if record["path"] in (1, 3)]
    assert all(record["failure_reason"] for record in failed)


# --- Test A: Root Negative Residual（Prompt H1-H5） --------------------------

def test_H1_large_negative_residual_rejected():
    # H1: G_norm=-0.2, tolerance=1e-5 必须拒绝（带符号比较会错误通过）。
    assert stp._candidate_residual_accepted(-0.2, 1e-5) is False


def test_H2_large_positive_residual_rejected():
    # H2: G_norm=+0.2 必须拒绝。
    assert stp._candidate_residual_accepted(0.2, 1e-5) is False


def test_H3_small_negative_residual_accepted():
    # H3: G_norm=-5e-6, tolerance=1e-5 可以接受。
    assert stp._candidate_residual_accepted(-5e-6, 1e-5) is True
    # NaN/Inf 一律拒绝。
    assert stp._candidate_residual_accepted(float("nan"), 1e-5) is False
    assert stp._candidate_residual_accepted(float("inf"), 1e-5) is False
    assert stp._candidate_residual_accepted(float("-inf"), 1e-5) is False


def test_H4_near_belief_large_negative_candidate_cannot_shadow_valid_root(
    tmp_path, patch_evaluators, monkeypatch
):
    """候选 A（大负残差、近 belief）必须被拒绝，B（严格容差、远）必须被选中。"""
    import sys as _sys_mod

    monkeypatch.setitem(
        _sys_mod.modules[__name__].PATH_ECONOMICS,
        0,
        {
            "regions": [
                (-0.8, 0.35, 1, 0),      # 真根 B: 2*exp(h)=0.35 -> h=-1.743
                (0.5, -196.7, 2, 0),     # h<0.5 侧 G 很大为正
                (float("inf"), 20.0, 3, 2),  # h>=0.5 侧 G 为负 -> 跳变伪候选 A
            ],
            "floor": False,
        },
    )
    sim = _sim(tmp_path, 1, [0.3])
    state = _state(sim)
    diagnostics = stp._calibrate_initial_hstar_batched(sim, state)

    assert diagnostics["success_count"] == 1
    audit = json.loads(
        (Path(tmp_path) / "initial_hstar_audit_4.json").read_text(encoding="utf-8")
    )
    record = audit["results"][0]
    assert record["terminal_status"] == "ROOT_FOUND"
    # B 不能被 A 遮蔽：唯一合法根是 B。
    assert record["root_count"] == 1
    assert record["selected_root"] == pytest.approx(-1.7430, abs=1e-2)
    assert record["selected_root"] < -1.0
    # A（跳变伪候选）被明确拒绝并记录原因。
    assert any(
        "RESIDUAL" in reason or "INFEASIBLE" in reason or "NONFINITE" in reason
        for reason in record["candidate_rejection_reasons"]
    )
    assert float(state["hatcf"][0]) == pytest.approx(-1.7430, abs=1e-2)


def test_H5_jump_sign_change_without_valid_root_is_not_root_found(
    tmp_path, patch_evaluators, monkeypatch
):
    """正负端点符号变化仅来自跳跃不连续时，不得宣称 ROOT_FOUND。"""
    import sys as _sys_mod

    monkeypatch.setitem(
        _sys_mod.modules[__name__].PATH_ECONOMICS,
        0,
        {
            "regions": [
                (0.5, -196.7, 2, 0),
                (float("inf"), 20.0, 3, 2),
            ],
            "floor": False,
        },
    )
    sim = _sim(tmp_path, 1, [0.3])
    state = _state(sim)
    with pytest.raises(RuntimeError, match="no valid clearing root"):
        stp._calibrate_initial_hstar_batched(sim, state)
    audit = json.loads(
        (Path(tmp_path) / "initial_hstar_audit_4.json").read_text(encoding="utf-8")
    )
    record = audit["results"][0]
    assert not record["terminal_status"].startswith("ROOT_FOUND")
    assert record["terminal_status"].startswith("NO_ROOT")
    # 带符号 bug 会让跳变伪候选通过验证并走到 FP64_REPLAY 失败分支；
    # 修复后必须在候选验证阶段就被拒绝（ALL_CANDIDATES_REJECTED）。
    assert record["failure_reason"] == "ALL_CANDIDATES_REJECTED"
    assert record["valid_root_count"] == 0
