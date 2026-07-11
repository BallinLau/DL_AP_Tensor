"""
SDF & FC1 模型

SDF: 随机贴现因子网络
FC1: 宏观状态预测网络（FC1_C 预测 ĉf，FC1_K 预测 ln Kf）

关键口径：
- FC1 直接在 physical scale 上预测增量与水平
- 不再依赖 scaler / inverse_transform
- df 中保存的一律是 physical scale

树状分支结构（支持任意 N 个分支）：
- parent: (w_t, k_t, c_t) → t 期父节点
- children: [(w_{t+1}^{(j)}, k_{t+1}^{(j)}, c_{t+1}^{(j)})] → N 条模拟路径
- 每条路径中 η 通过伯努利分布随机抽取

SDF 计算（对每条路径 j）：
    M^{(j)} = β^κ * exp((k_{t+1}^{(j)} - k_t)*(-γ) - κ/σ*(c_{t+1}^{(j)} - c_t))
              * (w_{t+1}^{(j)} / (w_t - exp(c_t)))^(κ-1)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Union
from .base import MLP

import sys
sys.path.append('..')
from config import Config


def compute_sdf(
    w_parent: torch.Tensor,
    w_children: Union[List[torch.Tensor], torch.Tensor],
    k_parent: torch.Tensor,
    k_children: Union[List[torch.Tensor], torch.Tensor],
    c_parent: torch.Tensor,
    c_children: Union[List[torch.Tensor], torch.Tensor],
    beta: float = None,
    gamma: float = None,
    kappa: float = None,
    sigma: float = None,
    eps: float = 1e-3,
    exponent_clip: Optional[float] = None
) -> Union[List[torch.Tensor], torch.Tensor]:
    """
    计算 SDF（支持任意数量的分支路径）
    
    M^{(j)} = β^κ * exp((k_{t+1}^{(j)} - k_t)*(-γ) - κ/σ*(c_{t+1}^{(j)} - c_t))
              * (w_{t+1}^{(j)} / (w_t - exp(c_t)))^(κ-1)
    
    Args:
        w_parent: (batch,) 父节点价值函数 w_t
        w_children: List[(batch,)] 或 (batch, n_branches) 或 (batch, n_branches, 1)
        k_parent: (batch,) 父节点资本对数 ln K_t
        k_children: List[(batch,)] 或 (batch, n_branches) 或 (batch, n_branches, 1)
        c_parent: (batch,) 父节点消费对数 ĉ_t
        c_children: List[(batch,)] 或 (batch, n_branches) 或 (batch, n_branches, 1)
        beta, gamma, kappa, sigma: 模型参数（默认从 Config 获取）
        eps: 数值稳定项
        exponent_clip: 对指数项进行裁剪，避免 exp 溢出
    
    Returns:
        M_list 或 M_tensor: List[torch.Tensor] 或 (batch, n_branches) 张量
    """
    # 从配置获取默认参数
    if beta is None:
        beta = Config.BETA
    if gamma is None:
        gamma = Config.GAMMA
    if kappa is None:
        kappa = Config.KAPPA
    if sigma is None:
        sigma = Config.SIGMA
    if exponent_clip is None:
        exponent_clip = getattr(Config, "SDF_EXPONENT_CLAMP", None)

    # β^κ
    tmp = beta ** kappa

    # 张量路径（批量分支）
    if isinstance(w_children, torch.Tensor):
        w_children_t = w_children
        k_children_t = k_children
        c_children_t = c_children

        if w_children_t.dim() == 3 and w_children_t.size(-1) == 1:
            w_children_t = w_children_t.squeeze(-1)
            k_children_t = k_children_t.squeeze(-1)
            c_children_t = c_children_t.squeeze(-1)
        if w_children_t.dim() == 1:
            w_children_t = w_children_t.unsqueeze(-1)
            k_children_t = k_children_t.unsqueeze(-1)
            c_children_t = c_children_t.unsqueeze(-1)

        w_parent_t = w_parent.squeeze(-1) if w_parent.dim() > 1 else w_parent
        k_parent_t = k_parent.squeeze(-1) if k_parent.dim() > 1 else k_parent
        c_parent_t = c_parent.squeeze(-1) if c_parent.dim() > 1 else c_parent

        denom = (w_parent_t - torch.exp(c_parent_t)).clamp_min(eps).unsqueeze(-1)
        ratio = (w_children_t / denom).clamp_min(eps)

        exponent = (
            (k_children_t - k_parent_t.unsqueeze(-1)) * (-gamma)
            - kappa / sigma * (c_children_t - c_parent_t.unsqueeze(-1))
        )
        if exponent_clip is not None:
            exponent = exponent.clamp(min=-exponent_clip, max=exponent_clip)

        M = torch.exp(exponent) * torch.pow(ratio, kappa - 1) * tmp

        return M

    # List 路径（逐分支）
    # 当期"财富-消费"
    denom = (w_parent - torch.exp(c_parent)).clamp_min(eps)

    # 对每条路径计算 M
    M_list = []
    for w_child, k_child, c_child in zip(w_children, k_children, c_children):
        ratio = (w_child / denom).clamp_min(eps)
        exponent = (
            (k_child - k_parent) * (-gamma) - kappa / sigma * (c_child - c_parent)
        )
        if exponent_clip is not None:
            exponent = exponent.clamp(min=-exponent_clip, max=exponent_clip)
        M = torch.exp(exponent) * torch.pow(ratio, kappa - 1) * tmp
        M_list.append(M)

    return M_list


def compute_sdf_legacy(
    w: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    kf: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cf: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    beta: float = None,
    gamma: float = None,
    kappa: float = None,
    sigma: float = None,
    eps: float = 1e-3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 SDF（旧版接口，兼容2分支情况）
    
    Args:
        w: (w1, w2, w3) = (w_parent, w_child1, w_child2)
        kf: (k1, k2, k3) = (k_parent, k_child1, k_child2)
        cf: (c1, c2, c3) = (c_parent, c_child1, c_child2)
    
    Returns:
        M1, M2: 两条路径的 SDF
    """
    w1, w2, w3 = w
    k1, k2, k3 = kf
    c1, c2, c3 = cf
    
    M_list = compute_sdf(
        w_parent=w1,
        w_children=[w2, w3],
        k_parent=k1,
        k_children=[k2, k3],
        c_parent=c1,
        c_children=[c2, c3],
        beta=beta,
        gamma=gamma,
        kappa=kappa,
        sigma=sigma,
        eps=eps
    )
    
    return M_list[0], M_list[1]


class SDFModel(nn.Module):
    """
    随机贴现因子 (SDF) 网络
    
    用于学习 M_{t,t+1}，满足 Euler 方程约束
    E_t[M_{t,t+1} R_{t+1}] = 1
    
    支持任意数量的分支路径：
        M^{(j)} = β^κ * exp((k_{t+1}^{(j)} - k_t)*(-γ) - κ/σ*(c_{t+1}^{(j)} - c_t))
                  * (w_{t+1}^{(j)} / (w_t - exp(c_t)))^(κ-1)
    """
    
    def __init__(
        self,
        input_dim: int = 4,  # (c_t, c_{t+1}, k_t, k_{t+1}) 或其它配置
        hidden_dims: List[int] = None,
        output_positive: bool = True
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = Config.SDF_HIDDEN_DIMS
        
        # SDF 主网络
        self.network = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            output_activation='softplus' if output_positive else None
        )
        
        # 用于计算 M 的参数
        self.beta = Config.BETA
        self.gamma = Config.GAMMA
        self.kappa = Config.KAPPA
        self.sigma = Config.SIGMA
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 SDF
        
        Args:
            x: 输入特征
        Returns:
            M: SDF 值
        """
        return self.network(x)
    
    def get_M(
        self,
        w_parent: torch.Tensor,
        w_children: Union[List[torch.Tensor], torch.Tensor],
        k_parent: torch.Tensor,
        k_children: Union[List[torch.Tensor], torch.Tensor],
        c_parent: torch.Tensor,
        c_children: Union[List[torch.Tensor], torch.Tensor],
        eps: float = 1e-3
    ) -> List[torch.Tensor]:
        """
        根据价值函数和状态变量计算 SDF（支持任意分支数）
        
        Args:
            w_parent: (batch,) 父节点价值函数
            w_children: List[(batch,)] 或 (batch, n_branches) 子节点价值函数
            k_parent: (batch,) 父节点资本对数
            k_children: 子节点资本对数
            c_parent: (batch,) 父节点消费对数
            c_children: 子节点消费对数
            eps: 数值稳定项
        
        Returns:
            M_list: List[torch.Tensor] - 每条路径的 SDF
        """
        return compute_sdf(
            w_parent=w_parent,
            w_children=w_children,
            k_parent=k_parent,
            k_children=k_children,
            c_parent=c_parent,
            c_children=c_children,
            beta=self.beta,
            gamma=self.gamma,
            kappa=self.kappa,
            sigma=self.sigma,
            eps=eps
        )
    
    def get_M_legacy(
        self,
        w: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        kf: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        cf: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        eps: float = 1e-3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        旧版接口，兼容2分支情况
        
        Args:
            w: (w1, w2, w3) = (w_parent, w_child1, w_child2)
            kf: (k1, k2, k3)
            cf: (c1, c2, c3)
        
        Returns:
            M1, M2
        """
        return compute_sdf_legacy(
            w, kf, cf,
            beta=self.beta,
            gamma=self.gamma,
            kappa=self.kappa,
            sigma=self.sigma,
            eps=eps
        )
    
    def compute_sdf_from_consumption(
        self, 
        c_t: torch.Tensor, 
        c_t1: torch.Tensor,
        k_t: torch.Tensor,
        k_t1: torch.Tensor
    ) -> torch.Tensor:
        """
        根据消费增长计算简单 SDF (CRRA 形式)
        
        标准 CRRA 形式：M = β * (C_{t+1}/C_t)^{-γ}
        """
        # 对数消费增长
        log_c_growth = torch.log(c_t1 + 1e-8) - torch.log(c_t + 1e-8)
        
        # CRRA SDF
        M = self.beta * torch.exp(-self.gamma * log_c_growth)
        
        return M


class FC1Model(nn.Module):
    """
    FC1 模型：宏观状态预测
    
    包含两个子网络（增量形式）：
    - FC1_C: 预测 Δĉf
    - FC1_K: 预测 Δln Kf
    
    输入维度: 4 = (x_{t-1}, x_t, ĉf_{t-1}, ln Kf_{t-1})
    """
    
    def __init__(
        self,
        input_dim: int = None,
        hidden_dims: List[int] = None
    ):
        super().__init__()
        
        if input_dim is None:
            input_dim = Config.FC1_INPUT_DIM
        if hidden_dims is None:
            hidden_dims = Config.FC1_HIDDEN_DIMS
        
        # FC1_C: 直接在物理尺度预测 Δĉf
        self.FC1_C = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1
        )
        
        # FC1_K: 直接在物理尺度预测 Δln Kf
        self.FC1_K = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1
        )
    
    def fit_scalers(
        self, 
        x: torch.Tensor, 
        hatcf: torch.Tensor, 
        lnkf: torch.Tensor
    ):
        """
        兼容旧接口。FC1 已不再使用 scaler，因此这里是 no-op。
        
        Args:
            x: 输入特征
            hatcf: 兼容保留
            lnkf: 兼容保留
        """
        return None
    
    def forward(
        self, 
        x: torch.Tensor,
        return_physical: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: (batch, 4) - (x_{t-1}, x_t, ĉf_{t-1}, ln Kf_{t-1})
            return_physical: 兼容保留。FC1 始终返回物理尺度。
        
        Returns:
            hatcf: (batch, 1) - 预测的 ĉf_{t+1}
            lnkf: (batch, 1) - 预测的 ln Kf_{t+1}
        """
        if x.shape[-1] < 4:
            raise ValueError("FC1 input must include (x_prev, x_curr, hatcf_prev, lnkf_prev)")

        delta_hatcf = self.FC1_C(x)
        delta_lnkf = self.FC1_K(x)

        # 采用增量建模：y_{t+1} = y_t + Δy
        hatcf_prev = x[..., 2:3]
        lnkf_prev = x[..., 3:4]
        hatcf = hatcf_prev + delta_hatcf
        lnkf = lnkf_prev + delta_lnkf

        return hatcf, lnkf
    
    def forward_normalized(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """兼容旧接口。FC1 不再区分 normalized / physical。"""
        return self.forward(x, return_physical=False)
    
    def forward_physical(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回物理尺度输出。"""
        return self.forward(x, return_physical=True)


class SDFFC1Combined(nn.Module):
    """
    SDF & FC1 组合模型
    
    统一管理 SDF 和 FC1 的训练与推理
    """
    # comment: 这个模型的核心输入是 (x_{t-1}, x_t, ĉf_{t-1}, lnKf_{t-1})，
    #          先用 FC1 输入为 (x_{t-1}, x_t, ĉf_{t-1}, lnKf_{t-1}) 预测 ĉf_t, lnKf_t，
    #          再用 ValueFunctionW 计算 w_{t-1}, w_t，最后用 w_{t-1}, w_t 计算 M(t-1, t)。
    def __init__(
        self,
        sdf_input_dim: int = 4,
        fc1_input_dim: int = 4,
        sdf_hidden_dims: List[int] = None,
        fc1_hidden_dims: List[int] = None,
        w_hidden_dims: List[int] = None
    ):
        super().__init__()
        
        self.sdf_model = SDFModel(
            input_dim=sdf_input_dim,
            hidden_dims=sdf_hidden_dims
        )
        
        self.fc1_model = FC1Model(
            input_dim=fc1_input_dim,
            hidden_dims=fc1_hidden_dims
        )
        
        self.value_model = ValueFunctionW(
            input_dim=3,
            hidden_dims=w_hidden_dims
        )
    
    @property
    def FC1_C(self):
        return self.fc1_model.FC1_C
    
    @property
    def FC1_K(self):
        return self.fc1_model.FC1_K
    
    def get_M_legacy(
        self,
        w: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        kf: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        cf: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        eps: float = 1e-3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """旧版接口，兼容2分支情况"""
        return self.sdf_model.get_M_legacy(w, kf, cf, eps)
    
    def forward_fc1(
        self, 
        x: torch.Tensor,
        return_physical: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测宏观状态（输入: x_{t-1}, x_t, ĉf_{t-1}, lnKf_{t-1}）"""
        return self.fc1_model(x, return_physical)
    
    def fit_fc1_scalers(
        self, 
        x: torch.Tensor, 
        hatcf: torch.Tensor, 
        lnkf: torch.Tensor
    ):
        """拟合 FC1 标准化参数"""
        self.fc1_model.fit_scalers(x, hatcf, lnkf)

    def forward_step(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        hatcf_prev: torch.Tensor,
        lnkf_prev: torch.Tensor,
        return_physical: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用 (x_{t-1}, x_t, ĉf_{t-1}, lnKf_{t-1}) 预测 ĉf_t、lnKf_t 并计算 M(t-1,t)。
        
        Returns:
            w_prev, w_curr, M, hatcf_curr, lnkf_curr
        """
        def to_col(t: torch.Tensor) -> torch.Tensor:
            return t if t.dim() > 1 else t.unsqueeze(-1)
        
        x_prev = to_col(x_prev)
        x_curr = to_col(x_curr)
        hatcf_prev = to_col(hatcf_prev)
        lnkf_prev = to_col(lnkf_prev)

        # 多分支批量路径：(batch, n_children, 1)
        if x_curr.dim() == 3:
            batch, n_children, _ = x_curr.shape

            x_prev_base = x_prev[:, 0, :] if x_prev.dim() == 3 else x_prev
            hatcf_prev_base = hatcf_prev[:, 0, :] if hatcf_prev.dim() == 3 else hatcf_prev
            lnkf_prev_base = lnkf_prev[:, 0, :] if lnkf_prev.dim() == 3 else lnkf_prev

            if x_prev.dim() == 2:
                x_prev = x_prev.unsqueeze(1).expand(-1, n_children, -1)
            if hatcf_prev.dim() == 2:
                hatcf_prev = hatcf_prev.unsqueeze(1).expand(-1, n_children, -1)
            if lnkf_prev.dim() == 2:
                lnkf_prev = lnkf_prev.unsqueeze(1).expand(-1, n_children, -1)

            fc1_input = torch.cat([x_prev, x_curr, hatcf_prev, lnkf_prev], dim=-1)
            flat = fc1_input.reshape(-1, fc1_input.shape[-1])
            hatcf_curr, lnkf_curr = self.forward_fc1(flat, return_physical=return_physical)
            hatcf_curr = hatcf_curr.view(batch, n_children, 1)
            lnkf_curr = lnkf_curr.view(batch, n_children, 1)

            w_prev = self.value_model(torch.cat([x_prev_base, hatcf_prev_base, lnkf_prev_base], dim=-1))
            w_curr_input = torch.cat([x_curr, hatcf_curr, lnkf_curr], dim=-1)
            w_curr = self.value_model(w_curr_input.reshape(-1, w_curr_input.shape[-1]))
            w_curr = w_curr.view(batch, n_children, 1)

            M = self.sdf_model.get_M(
                w_parent=w_prev.squeeze(-1),
                w_children=w_curr.squeeze(-1),
                k_parent=lnkf_prev_base.squeeze(-1),
                k_children=lnkf_curr.squeeze(-1),
                c_parent=hatcf_prev_base.squeeze(-1),
                c_children=hatcf_curr.squeeze(-1)
            )
            if isinstance(M, list):
                M = M[0]
            if M.dim() == 2:
                M = M.unsqueeze(-1)

            return w_prev, w_curr, M, hatcf_curr, lnkf_curr

        # 单分支路径：(batch, 1)
        fc1_input = torch.cat([x_prev, x_curr, hatcf_prev, lnkf_prev], dim=-1)
        hatcf_curr, lnkf_curr = self.forward_fc1(fc1_input, return_physical=return_physical)

        w_prev = self.value_model(torch.cat([x_prev, hatcf_prev, lnkf_prev], dim=-1))
        w_curr = self.value_model(torch.cat([x_curr, hatcf_curr, lnkf_curr], dim=-1))

        M_list = self.sdf_model.get_M(
            w_parent=w_prev,
            w_children=[w_curr],
            k_parent=lnkf_prev,
            k_children=[lnkf_curr],
            c_parent=hatcf_prev,
            c_children=[hatcf_curr]
        )

        return w_prev, w_curr, M_list[0], hatcf_curr, lnkf_curr


class ValueFunctionW(nn.Module):
    """
    价值函数 W 网络
    
    用于 SDF Loss 中的价值函数部分
    输出 w = (w_t, w_{t+1}^0, w_{t+1}^1, ...)
    """
    
    def __init__(
        self,
        input_dim: int = 2,  # (c, k) 或类似
        hidden_dims: List[int] = None,
        output_positive: bool = True,
        surplus_floor: float = None
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        self.network = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            output_activation='softplus' if output_positive else None
        )
        self.surplus_floor = (
            float(surplus_floor)
            if surplus_floor is not None
            else float(getattr(Config, "W_SURPLUS_FLOOR", 1e-3))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            w: (batch, 1) - 价值函数值
        """
        # 结构化参数化：w = exp(c) + surplus，确保 w - exp(c) 始终为正且有下界
        # x 约定为 [x_macro, hatcf(c), lnkf(k)]
        c = x[..., 1:2]
        surplus = self.network(x) + self.surplus_floor
        return torch.exp(c) + surplus
