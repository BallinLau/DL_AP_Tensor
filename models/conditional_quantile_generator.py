"""冻结的条件分位数生成器：初始 b 分布实验设计工具。

来源：b_distribution_network_preview_20261109_v1/b_distribution_network_preview_v1.ipynb
（2026-11-09 预览实验，8 维分布编码 -> 小型 MLP -> 单调有理二次样条分位数映射）。
本模块把该原型整理为可复用类，供训练管线在"从某 episode 续训 + 不同初始 b 分布"
实验中采样初始杠杆 b。

实验设计语义（必须保持）：
- 固定编码 c_v = 固定一个分布；b⁰(v,p,j) = Q_θ(u_p,j ; c_v)，u ~ U(0,1)。
- 换 u 是同一分布内换企业；换 c_v 才是换分布。
- 共享随机数（配对设计）：同一集内所有版本用相同的 U；同一集内多次重模拟
  也用相同的 U（确定性重放）；不同集用新 U（由编排方 set_seed(base + ep)）。
- 生成器永远冻结：不进任何优化器、不接受经济损失梯度。
- 本分布只含连续部分；b=0 点质量留作后续扩展。
- 分布是人为构造的实验设计分布，不是从数据估计的真实经济分布。

训练接入点：data/simulate_ts.py::SimulateTS._sample_initial_b 的 callable 分支
（models['dist_b']），不改经济模型网络结构。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence, Union

import torch
from torch import nn
import torch.nn.functional as F


class ConditionalQuantileNet(nn.Module):
    """条件单调 RQS 分位数网络 Q(u; c)，与预览 notebook 的结构/前向逐位一致。

    结构：code_dim -> hidden_dims(96,96, SiLU) -> 2*bins+1；
    输出解析为样条区间宽度（softmax + ε 平滑）与节点导数（softplus + ε），
    端点固定 Q(0)=0、Q(1)=1，且 dQ/du > 0 严格单调。
    """

    def __init__(
        self,
        bins: int = 32,
        code_dim: int = 8,
        hidden_dims: Sequence[int] = (96, 96),
    ):
        super().__init__()
        self.bins = int(bins)
        self.code_dim = int(code_dim)
        self.hidden_dims = [int(h) for h in hidden_dims]
        layers = []
        prev = self.code_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        layers.append(nn.Linear(prev, 2 * self.bins + 1))
        self.net = nn.Sequential(*layers)
        # 末层零初始化：与预览训练配方一致（起点接近均匀分位数映射）
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, u, c, return_derivative: bool = False):
        c = c.to(dtype=self.net[0].weight.dtype)
        raw = self.net(c / 3.0)
        h = 1.0e-4 + (1.0 - self.bins * 1.0e-4) * F.softmax(
            raw[:, : self.bins], dim=-1
        )
        d = 1.0e-3 + F.softplus(raw[:, self.bins :])
        y = torch.cat(
            [torch.zeros_like(h[:, :1]), torch.cumsum(h, dim=-1)], dim=-1
        )
        y = torch.cat([y[:, :-1], torch.ones_like(y[:, -1:])], dim=-1)
        if u.ndim == 1:
            u = u[None, :].expand(c.shape[0], -1)
        u = u.to(dtype=c.dtype, device=c.device)
        idx = (u * self.bins).long().clamp(0, self.bins - 1)
        theta = (u - idx.to(u.dtype) / self.bins) * self.bins
        height = torch.gather(h, 1, idx)
        y0 = torch.gather(y, 1, idx)
        dl = torch.gather(d, 1, idx)
        dr = torch.gather(d, 1, idx + 1)
        delta = height * self.bins
        cross = theta * (1.0 - theta)
        denom = delta + (dl + dr - 2.0 * delta) * cross
        out = y0 + height * (delta * theta.square() + dl * cross) / denom
        deriv = delta.square() * (
            dr * theta.square() + 2.0 * delta * cross + dl * (1.0 - theta).square()
        ) / denom.square()
        return (out, deriv) if return_derivative else out


def parse_distribution_code(
    distribution_code: Union[str, Sequence[float]]
) -> list:
    """解析分布编码：逗号分隔字符串或数值序列 -> float 列表。"""
    if isinstance(distribution_code, str):
        parts = [p.strip() for p in distribution_code.split(",") if p.strip()]
        if not parts:
            raise ValueError("distribution_code 字符串为空")
        return [float(p) for p in parts]
    return [float(v) for v in distribution_code]


def load_generator_checkpoint(
    checkpoint_path: Union[str, Path], device: Union[str, torch.device] = "cpu"
) -> ConditionalQuantileNet:
    """加载预览产物 conditional_quantile_generator.pt，恢复结构与权重。

    checkpoint 字段（预览 notebook 的保存格式）：
    state_dict / bins / code_dim / hidden_dims / seed / validation / purpose。
    预览权重为 float64（导出前做过 .double()），此处原样保留 double，
    与预览验证（严格单调性检查）口径一致。
    """
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing = [
        key
        for key in ("state_dict", "bins", "code_dim", "hidden_dims")
        if key not in payload
    ]
    if missing:
        raise ValueError(f"生成器 checkpoint 缺少字段: {missing}")
    net = ConditionalQuantileNet(
        bins=int(payload["bins"]),
        code_dim=int(payload["code_dim"]),
        hidden_dims=list(payload["hidden_dims"]),
    ).double()
    net.load_state_dict(payload["state_dict"], strict=True)
    net.eval()
    net.requires_grad_(False)
    return net.to(device)


class InitialBQuantileSampler(nn.Module):
    """SimulateTS.models['dist_b'] 的 callable 适配器（冻结）。

    调用契约（data/simulate_ts.py::_sample_initial_b 的 callable 分支）：
        b = sampler(n=n, z=z, device=device) -> 形状 (n,) 的初始杠杆。

    共享随机数语义（配对实验设计）：
    - 每次调用都以 self.seed 为种子、在 CPU 上确定性生成 u ~ U(0,1)，
      因此同一集内多次重模拟返回完全相同的 u（同 n 下逐元素一致）；
    - 不同版本只差 code、共享 base seed -> 同一 u 在各版本映射到各自分布的
      同一分位（配对企业）；
    - 不同集换新 U：编排方每集调用 set_seed(base_seed + episode)。

    z 参数仅为调用契约保留：初始 b 分布与 z 独立（实验设计决定）。
    生成器权重冻结（requires_grad=False），不注册进任何优化器。
    """

    def __init__(
        self,
        generator: ConditionalQuantileNet,
        distribution_code: Union[str, Sequence[float]],
        seed: int,
        lower: float = 0.0,
        upper: float = 1.0,
        generator_checkpoint: Union[str, Path, None] = None,
    ):
        super().__init__()
        code = parse_distribution_code(distribution_code)
        if len(code) != generator.code_dim:
            raise ValueError(
                f"distribution_code 长度 {len(code)} != 生成器 code_dim "
                f"{generator.code_dim}"
            )
        if not float(lower) < float(upper):
            raise ValueError(f"lower({lower}) 必须严格小于 upper({upper})")
        self.generator = generator
        self.generator.eval()
        self.generator.requires_grad_(False)
        self.register_buffer(
            "code",
            torch.tensor(code, dtype=torch.float64).reshape(1, -1),
            persistent=False,
        )
        self.seed = int(seed)
        self.seed_base = int(seed)
        self.lower = float(lower)
        self.upper = float(upper)
        self.generator_checkpoint = (
            str(generator_checkpoint) if generator_checkpoint is not None else None
        )

    def set_seed(self, seed: int) -> None:
        """刷新共享随机数种子（每集 base + episode）。"""
        self.seed = int(seed)

    @torch.no_grad()
    def forward(self, n: int, z=None, device=None, dtype=None):
        n = int(n)
        dev = next(self.generator.parameters()).device
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        u = torch.rand(n, generator=gen, dtype=torch.float64)
        q = self.generator(u.to(dev), self.code.detach().to(dev))
        b = self.lower + (self.upper - self.lower) * q[0].to(torch.float64)
        b = b.clamp(min=self.lower, max=self.upper)
        out = b.to(torch.float32)
        if dtype is not None:
            out = out.to(dtype)
        return out.to(device=device) if device is not None else out

    def state_dict(self, *args, **kwargs):
        # 冻结的实验设计工具：权重以生成器 checkpoint 文件为权威
        # （路径+sha256 记录于 metadata），不重复写入 combined.pt 模型载荷。
        return {}

    def describe(self) -> dict:
        """可 JSON 化的配置描述（日志 / run metadata 用）。"""
        info = {
            "source": "conditional_quantile_network",
            "frozen": True,
            "generator_checkpoint": self.generator_checkpoint,
            "bins": int(self.generator.bins),
            "code_dim": int(self.generator.code_dim),
            "hidden_dims": [int(h) for h in self.generator.hidden_dims],
            "distribution_code": [float(v) for v in self.code[0].tolist()],
            "seed_base": int(self.seed_base),
            "seed_current": int(self.seed),
            "lower": float(self.lower),
            "upper": float(self.upper),
        }
        if self.generator_checkpoint:
            path = Path(self.generator_checkpoint)
            if path.is_file():
                info["generator_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        return info


def build_initial_b_sampler(
    generator_checkpoint: Union[str, Path],
    distribution_code: Union[str, Sequence[float]],
    seed: int,
    lower: float = 0.0,
    upper: float = 1.0,
) -> InitialBQuantileSampler:
    """构建冻结采样器（训练管线唯一入口）。

    generator_checkpoint: 预览产物 conditional_quantile_generator.pt；
    distribution_code: 该训练版本的固定编码 c_v（逗号分隔字符串或数值序列）；
    seed: 共享随机数 base 种子（per-episode 种子 = seed + episode）；
    lower/upper: 输出区间（训练侧传 Config.SIM_B_INIT_MIN/MAX）。
    """
    net = load_generator_checkpoint(generator_checkpoint, device="cpu")
    return InitialBQuantileSampler(
        generator=net,
        distribution_code=distribution_code,
        seed=seed,
        lower=lower,
        upper=upper,
        generator_checkpoint=generator_checkpoint,
    )
