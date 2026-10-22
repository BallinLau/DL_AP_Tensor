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


# 每条 path 的分段 C_raw(h)：h<0 与 h>=0 两段，K 固定 2.0。
PATH_ECONOMICS = {
    0: {"c_neg": 0.4, "c_pos": 4.0, "floor": False},   # 双根 + h=0 跳变
    1: {"c_neg": 100.0, "c_pos": 100.0, "floor": False},  # 支撑区间内无变号
    2: {"c_neg": 0.8, "c_pos": 0.8, "floor": True},    # 单根 + floor
    3: {"c_neg": -1.0, "c_pos": -1.0, "floor": False},  # 处处经济不可行
}


def _c_raw(path: int, h: float) -> float:
    spec = PATH_ECONOMICS[path]
    return spec["c_neg"] if h < 0.0 else spec["c_pos"]


def _record(h: float, path: int) -> dict:
    c = _c_raw(path, h)
    k = 2.0
    g = torch.exp(torch.tensor(h, dtype=torch.float64)) * k - c
    scale = max(1.0, abs(c), abs(k))
    finite = torch.isfinite(torch.tensor(h)) and torch.isfinite(torch.tensor(c))
    feasible = finite and c > 0.0 and k > 0.0
    floor_flag = bool(PATH_ECONOMICS[path]["floor"])
    h_region = h < 0.0
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
        "inv_count": torch.tensor(1 if h_region else 2, dtype=torch.long),
        "def_count": torch.tensor(0 if h_region else 1, dtype=torch.long),
        "entry_accepted": torch.tensor(0.0, dtype=torch.float32),
    }


def _fake_rows(sim, base_state, rows, h_rows, *, rng_cpu=None, rng_cuda=None,
               return_state=False, **kwargs):
    records = []
    failures = {}
    for index in range(int(rows.numel())):
        path = int(rows[index])
        records.append(_record(float(h_rows[index]), path))
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
            record = _record(float(h_grid[row, column]), path)
            for name in names:
                packed[name][row] = record[name]
        for name in names:
            columns[name].append(packed[name])
        failures_list.append({})
    return {name: torch.stack(columns[name], dim=1) for name in names}, failures_list


def _sim(tmp_path, n_paths, beliefs) -> "_Sim":
    return _Sim(tmp_path, n_paths, beliefs)


class _Sim(SimpleNamespace):
    def __init__(self, tmp_path, n_paths, beliefs):
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
        {"c_neg": 0.8, "c_pos": 0.8, "floor": False},
    )
    sim = _sim(tmp_path, 2, [-2.0, 0.0])
    state = _state(sim)
    diagnostics = stp._calibrate_initial_hstar_batched(sim, state)

    assert diagnostics["success_count"] == 2
    assert diagnostics["root_solver"] == "strict_global_batched_production_node_operator"
    # path0 两个有效根 + 一个跳变区间；选择离 belief(-2) 更近的低根。
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
    assert by_path[0]["selected_root"] == pytest.approx(min(roots))
    assert by_path[1]["terminal_status"] == "ROOT_FOUND"
    # 正式状态写入所选根。
    assert float(state["hatcf"][0]) == pytest.approx(-1.6094, abs=1e-4)
    assert float(state["hatcf"][1]) == pytest.approx(-0.9163, abs=1e-4)
    assert state["lnkf"].isfinite().all()


def test_strict_global_floor_recorded_at_root(tmp_path, patch_evaluators):
    # 单条 path（经济 2）带 consumption-floor；raw 出清根必须正常找到，
    # floor flag 只记录在审计里，不阻塞 root。
    sim = _sim(tmp_path, 1, [0.0])
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
