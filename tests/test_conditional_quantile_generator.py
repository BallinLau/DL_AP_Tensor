"""models/conditional_quantile_generator.py 的 CPU 冒烟测试。

覆盖：
- 预览 checkpoint 格式的加载/往返一致性；
- Q(0)=0、Q(1)=1、严格单调、导数为正；
- 共享随机数配对语义（同 seed 同 U：不同编码下同一索引企业落在同一分位；
  同编码多次调用确定性重放；set_seed 换 U）；
- SimulateTS._sample_initial_b 的 callable 钩子契约与 uniform 回退不变性。
"""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch  # noqa: E402

from data.simulate_ts import SimulateTS  # noqa: E402
from models.conditional_quantile_generator import (  # noqa: E402
    ConditionalQuantileNet,
    build_initial_b_sampler,
    load_generator_checkpoint,
    parse_distribution_code,
)


def _make_checkpoint(tmp_path, bins=16, code_dim=8, hidden=(32, 32)):
    torch.manual_seed(7)
    net = ConditionalQuantileNet(
        bins=bins, code_dim=code_dim, hidden_dims=hidden
    ).double()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    payload = {
        "state_dict": net.state_dict(),
        "bins": bins,
        "code_dim": code_dim,
        "hidden_dims": list(hidden),
        "seed": 0,
        "validation": {"heldout_mean_W1_grid": 0.0},
        "purpose": "test",
    }
    path = tmp_path / "generator.pt"
    torch.save(payload, path)
    return path, net


def test_parse_distribution_code():
    assert parse_distribution_code("0.5,-1,2.25") == [0.5, -1.0, 2.25]
    assert parse_distribution_code([1, 2.5]) == [1.0, 2.5]


def test_load_generator_checkpoint_round_trip(tmp_path):
    path, ref_net = _make_checkpoint(tmp_path)
    net = load_generator_checkpoint(path)
    assert net.bins == 16 and net.code_dim == 8 and net.hidden_dims == [32, 32]
    u = torch.linspace(0.0, 1.0, 65, dtype=torch.float64)
    c = torch.randn(5, 8, dtype=torch.float64)
    with torch.no_grad():
        expected = ref_net(u, c)
        got = net(u, c)
    assert torch.equal(expected, got)  # float64 逐位一致
    assert all(not p.requires_grad for p in net.parameters())


def test_quantile_properties(tmp_path):
    path, _ = _make_checkpoint(tmp_path)
    net = load_generator_checkpoint(path)
    u = torch.linspace(0.0, 1.0, 2049, dtype=torch.float64)
    c = torch.randn(6, 8, dtype=torch.float64) * 3.0
    with torch.no_grad():
        q, deriv = net(u, c, return_derivative=True)
    assert torch.allclose(q[:, 0], torch.zeros(6, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(q[:, -1], torch.ones(6, dtype=torch.float64), atol=1e-9)
    assert (torch.diff(q, dim=-1) > 0).all()  # 严格单调
    assert (deriv > 0).all()


def test_sampler_determinism_and_pairing(tmp_path):
    path, _ = _make_checkpoint(tmp_path)
    code_a = "0,2,0,0,1.5,0,0,0"
    code_b = "1,-1,0.5,0,-1,0,0,0"
    sa = build_initial_b_sampler(path, code_a, seed=42)
    sb = build_initial_b_sampler(path, code_b, seed=42)
    sa2 = build_initial_b_sampler(path, code_a, seed=42)

    b_a1 = sa(n=500, z=None, device=torch.device("cpu"))
    b_a2 = sa(n=500, z=None, device=torch.device("cpu"))
    b_a3 = sa2(n=500, z=None, device=torch.device("cpu"))
    b_b1 = sb(n=500, z=None, device=torch.device("cpu"))

    assert b_a1.shape == (500,)
    assert b_a1.dtype == torch.float32
    assert torch.equal(b_a1, b_a2)  # 同集内多次重模拟 -> 相同 U
    assert torch.equal(b_a1, b_a3)  # 同编码同 seed 的另一"版本"实例 -> 相同 U
    assert not torch.equal(b_a1, b_b1)  # 换编码才是换分布
    # 配对：两版本的 b 序由同一批 u 驱动，秩序一致（同一索引=同一分位）
    assert torch.equal(torch.argsort(b_a1), torch.argsort(b_b1))
    assert float(b_a1.min()) >= 0.0 and float(b_a1.max()) <= 1.0

    sa.set_seed(43)  # 不同集换新 U
    b_a4 = sa(n=500, z=None, device=torch.device("cpu"))
    assert not torch.equal(b_a1, b_a4)


def test_state_dict_empty_and_describe(tmp_path):
    path, _ = _make_checkpoint(tmp_path)
    sampler = build_initial_b_sampler(
        path, [0.0] * 8, seed=7, lower=0.0, upper=1.0
    )
    assert sampler.state_dict() == {}  # 冻结工具不进模型载荷
    assert all(not p.requires_grad for p in sampler.parameters())
    info = sampler.describe()
    assert info["source"] == "conditional_quantile_network"
    assert info["frozen"] is True
    assert info["distribution_code"] == [0.0] * 8
    assert info["generator_sha256"]


def test_sample_initial_b_hook_contract(tmp_path):
    path, _ = _make_checkpoint(tmp_path)
    sampler = build_initial_b_sampler(path, "0,2,0,0,1.5,0,0,0", seed=11)
    stub = SimpleNamespace(
        models={"dist_b": sampler},
        config=SimpleNamespace(SIM_B_INIT_MIN=0.0, SIM_B_INIT_MAX=1.0),
    )
    z = torch.zeros(40)
    hooked = SimulateTS._sample_initial_b(stub, 40, z, torch.device("cpu"))
    direct = sampler(n=40, z=z, device=torch.device("cpu"))
    assert torch.equal(hooked.squeeze(-1), direct)  # callable 分支逐位一致

    # dist_b=None -> 原 uniform 回退（行为不变性）
    stub_uniform = SimpleNamespace(
        models={"dist_b": None},
        config=SimpleNamespace(SIM_B_INIT_MIN=0.2, SIM_B_INIT_MAX=0.8),
    )
    b_u = SimulateTS._sample_initial_b(stub_uniform, 400, z, torch.device("cpu"))
    assert b_u.shape == (400,)
    assert float(b_u.min()) >= 0.2 and float(b_u.max()) <= 0.8


def test_build_initial_b_sampler_validates_inputs(tmp_path):
    path, _ = _make_checkpoint(tmp_path, code_dim=8)
    try:
        build_initial_b_sampler(path, "1,2,3", seed=0)  # 长度错误
    except ValueError:
        pass
    else:
        raise AssertionError("code 长度不匹配应报 ValueError")
    try:
        build_initial_b_sampler(path, [0.0] * 8, seed=0, lower=1.0, upper=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("lower>=upper 应报 ValueError")
