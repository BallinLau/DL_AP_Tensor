"""t0 h-star 根抛光（_polish_initial_hstar_root）两段式加固的针对性测试。

覆盖修复引入的三类行为（验收门 G_norm/hatc 双单位语义与声明容差完全不变）：

- 慢收缩定点映射（|T'|≈0.97，40 轮预算内不达标）→ Phase 2 二分回退收敛；
- 发散/振荡定点映射（|T'|>1）→ 最优跟踪 + 二分回退收敛；
- 决策跳变处无真根 → 诚实失败（不伪造），并记录有限迭代轨迹；
- 原本一步收敛的良态映射行为不变（回归保护）。
"""

from pathlib import Path
from types import SimpleNamespace
import math
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import data.simulate_ts_parallel as stp  # noqa: E402

ROOT_TOL = 1e-5
POLISH_TARGET = 0.25 * ROOT_TOL
K_CONST = 2.0


def _record(c: float, h: float) -> dict:
    g = math.exp(h) * K_CONST - c
    scale = max(1.0, abs(c), abs(K_CONST))
    finite = math.isfinite(h) and math.isfinite(c)
    feasible = finite and c > 0.0
    return {
        "h": torch.tensor(h, dtype=torch.float32),
        "G": torch.tensor(float(g), dtype=torch.float32),
        "G_norm": torch.tensor(float(g) / scale, dtype=torch.float32),
        "C_raw": torch.tensor(float(c), dtype=torch.float32),
        "C_eff": torch.tensor(float(max(c, 1e-4)), dtype=torch.float32),
        "K": torch.tensor(K_CONST, dtype=torch.float32),
        "floor_applied": torch.tensor(False, dtype=torch.bool),
        "finite": torch.tensor(finite, dtype=torch.bool),
        "feasible": torch.tensor(feasible, dtype=torch.bool),
        "accounting_valid": torch.tensor(True, dtype=torch.bool),
        "resource_feasible": torch.tensor(feasible, dtype=torch.bool),
        "resource_residual": torch.tensor(0.0, dtype=torch.float32),
        "n_firms": torch.tensor(1.0, dtype=torch.float32),
        "inv_count": torch.tensor(1, dtype=torch.long),
        "def_count": torch.tensor(0, dtype=torch.long),
        "entry_accepted": torch.tensor(0.0, dtype=torch.float32),
    }


def _fake_rows(sim, base_state, rows, h_rows, *, rng_cpu=None, rng_cuda=None,
               return_state=False, **kwargs):
    del base_state, rng_cpu, rng_cuda
    records = []
    for index in range(int(rows.numel())):
        h_value = float(h_rows[index])
        records.append(_record(sim.cfun(h_value), h_value))
    return records, {}, {} if return_state else None


def _sim(cfun) -> SimpleNamespace:
    return SimpleNamespace(cfun=cfun)


def _state() -> dict:
    return {}


def _run_polish(cfun, h0: float, max_iterations: int = 40):
    sim = _sim(cfun)
    rows = torch.tensor([0], dtype=torch.long)
    h_rows = torch.tensor([h0], dtype=torch.float32)
    return stp._polish_initial_hstar_root(
        sim,
        _state(),
        rows,
        h_rows,
        rng_cpu=torch.get_rng_state(),
        rng_cuda=None,
        strict_tolerance=ROOT_TOL,
        root_tolerance=ROOT_TOL,
        max_iterations=max_iterations,
    )


def _gates_ok(cfun, h_value: float) -> bool:
    """与抛光验收门逐字一致：|G_norm|<=tol 且 |hatc_expected-h|<=0.25*tol。"""
    c = cfun(h_value)
    if not (math.isfinite(c) and c > 0.0):
        return False
    g = math.exp(h_value) * K_CONST - c
    scale = max(1.0, abs(c), abs(K_CONST))
    g_norm = g / scale
    hatc_expected = math.log(c / (K_CONST + 1e-8))
    return (
        abs(g_norm) <= ROOT_TOL
        and abs(hatc_expected - h_value) <= POLISH_TARGET
    )


def test_slow_contraction_map_converges_within_fallback_budget(monkeypatch):
    # T(h)=ln(1+0.97h)：|T'|=0.97，从 h0=3e-4 的 hatc 偏差 9e-6 降到 2.5e-6
    # 需要 ~43 轮 > 40 轮预算——精确复刻生产失败模式；二分回退必须收敛。
    monkeypatch.setattr(stp, "_initial_hstar_evaluate_rows", _fake_rows)
    cfun = lambda h: 2.0 * (1.0 + 0.97 * h)  # noqa: E731
    h, records, failures, done, iters, traj = _run_polish(cfun, 3e-4)
    assert bool(done[0]), "fallback 未收敛"
    assert _gates_ok(cfun, float(h[0])), "返回根未过双单位验收门"
    assert records[0] is not None
    assert abs(float(h[0])) <= 3e-4  # 收敛到根 0 附近


def test_divergent_fixed_point_converges_via_bisect_fallback(monkeypatch):
    # T(h)=ln(1+2.4h)：|T'|=1.2>1，全步定点发散；阻尼(1+0.2a)>1 仍发散。
    # 最优跟踪保住起点邻域，二分回退收敛到真根 h=0。
    monkeypatch.setattr(stp, "_initial_hstar_evaluate_rows", _fake_rows)
    cfun = lambda h: 2.0 * (1.0 + 2.4 * h)  # noqa: E731
    h, records, failures, done, iters, traj = _run_polish(cfun, 2e-5)
    assert bool(done[0]), "发散映射未收敛"
    assert _gates_ok(cfun, float(h[0])), "返回根未过双单位验收门"


def test_jump_without_true_root_fails_honestly_with_trajectory(monkeypatch):
    # G(h) 在 h=-1.5 处从 -0.0755 跳到 +0.0492（norm ~2.5e-2 ≫ tol），
    # 两段内部各自的真根都在本段之外——全局无根。必须诚实失败且不伪造。
    monkeypatch.setattr(stp, "_initial_hstar_evaluate_rows", _fake_rows)
    cfun = lambda h: 0.5 if h < -1.5 else 0.42  # noqa: E731
    h, records, failures, done, iters, traj = _run_polish(cfun, -1.5)
    assert not bool(done[0]), "无真根路径被伪造收敛"
    assert not _gates_ok(cfun, float(h[0])), "返回值不应满足验收门"
    assert traj.get(0), "失败路径必须携带迭代轨迹"
    assert math.isfinite(float(h[0]))


def test_fast_convergence_unchanged(monkeypatch):
    # 良态：T(h)=ln(1) = 0 恒等吸收，两步内收敛，行为与原实现一致。
    monkeypatch.setattr(stp, "_initial_hstar_evaluate_rows", _fake_rows)
    cfun = lambda h: 2.0  # noqa: E731
    h, records, failures, done, iters, traj = _run_polish(cfun, 1e-4)
    assert bool(done[0])
    assert iters <= 3
    assert _gates_ok(cfun, float(h[0]))
