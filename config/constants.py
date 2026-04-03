"""
配置常量模块
定义所有模型和训练所需的常量参数
"""

from enum import IntEnum
from typing import List, Tuple
import torch


class SIMMODEL(IntEnum):
    """
    Firm-state向量的列顺序枚举
    必须与训练时的输入格式完全一致
    s_i = (b, z, η, i, x, ĉf, ln Kf)
    """
    B = 0      # 杠杆/债务状态 (firm-level)
    Z = 1      # 生产率/冲击 (firm-level)
    ETA = 2    # 是否可再融资 (firm-level, {0,1})
    I = 3      # 投资成本/摩擦 (firm-level, continuous)
    X = 4      # 宏观状态 (macro-level)
    HATCF = 5  # 宏观消费相关状态 (macro-level)
    LNKF = 6   # 宏观资本对数 (macro-level)


class Config:
    """
    全局配置类
    包含所有模型、数据生成和训练所需的参数
    """
    
    # ========== 经济参数 ==========
    # AR(1) 宏观冲击参数
    RHO_X = 0.95          # 宏观生产率自相关系数
    SIGMA_X = 0.012       # 宏观冲击标准差
    XBAR = -2.0           # 宏观生产率均值
    
    # AR(1) 公司冲击参数
    RHO_Z = 0.90          # 公司生产率自相关系数
    SIGMA_Z = 0.36        # 公司冲击标准差
    ZBAR = 0.0            # 公司生产率均值
    
    # 折旧与增长
    DELTA = 0.02          # 折旧率
    G = 1.14              # 增长因子
    
    # 税率与成本
    TAU = 0.2             # 公司税率
    KAPPA_B = 0.004       # 债务融资成本
    KAPPA_E = 0.025       # 股权融资成本
    PHI = 0.4             # 破产成本参数
    
    # 投资与再融资
    I_THRESHOLD = 0.2     # 投资成本上限
    PV_I_INTEGRATION_POINTS = 11  # Vhat/P/chi/bar_z 对 i 的离散积分点数
    ZETA = 0.03           # 再融资可得性概率
    # 模拟阶段杠杆支持集（用于保持违约区域可识别）
    SIM_B_INIT_MIN = 0.0
    SIM_B_INIT_MAX = 1.0
    ENTRY_B_MIN = 0.0
    ENTRY_B_MAX = 1.0
    
    # SDF 参数
    BETA = 0.942          # 贴现因子
    GAMMA = 4.0           # 风险厌恶系数
    SIGMA = 2.0           # 消费波动系数（与 DL7 一致）
    KAPPA = -6.0          # 价值函数幂次 (1-GAMMA)/(1-1/SIGMA)
    SIGMA_C = 0.01        # 消费波动系数
    SDF_EXPONENT_CLAMP = 10.0  # SDF 指数项裁剪，防止早期训练数值爆炸
    # W 参数化下界：w = exp(c) + surplus，surplus >= W_SURPLUS_FLOOR
    W_SURPLUS_FLOOR = 1e-3
    # P/bar_z 平滑温度（避免硬阈值导致梯度死区）
    P_SOFTPLUS_BETA = 8.0
    BARZ_LOGIT_TEMP = 3.0
    
    # ========== 模型架构参数 ==========
    # 共享层维度 (INCREASED for higher memory usage)
    SHARE_LAYER_HIDDEN_DIMS = [256, 256, 128, 64]

    # 各 Head 维度 (INCREASED)
    Q_HEAD_DIMS = [64, 32, 16]
    BP0_HEAD_DIMS = [64, 32, 16]
    BPI_HEAD_DIMS = [64, 32, 16]
    P0_HEAD_DIMS = [64, 32, 16]
    PI_HEAD_DIMS = [64, 32, 16]

    # SDF & FC1 维度 (INCREASED)
    SDF_HIDDEN_DIMS = [128, 64, 32]
    SDF_INPUT_DIM = 4
    FC1_BASE_INPUT_DIM = 4
    FC1_SUMMARY_DIM = 4    # (b_mean_t, b_std_t, z_mean_t, z_std_t)
    FC1_INPUT_DIM = FC1_BASE_INPUT_DIM + FC1_SUMMARY_DIM
    FC1_HIDDEN_DIMS = [64, 32, 16]
    
    # FC2 维度
    FC2_INPUT_DIM = 201    # 100(b分位点) + 100(z分位点) + x
    FC2_HIDDEN_DIMS = [128, 64, 32]
    FC2_OUTPUT_DIM = 2     # (ĉ, ln K)
    
    # ========== 数据参数 ==========
    GROUP_SIZE = 2         # 每条path的公司数（sample模式）
    SIMULATE_GROUP_SIZE = 200  # 每条path的公司数（simulate模式）
    BRANCH_NUM = 2         # 分支数
    QUANTILE_NUM = 100     # 分位数数量
    
    # ========== 训练参数 ==========
    BATCH_SIZE = 4096
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-3
    MAX_GRAD_NORM = 1.0
    
    # Loss 权重
    LAMBDA_SDF = 1.0
    LAMBDA_FC = 1.0
    LAMBDA_TRANS = 0.5     # FC2 跨期一致性权重
    
    # AIO 权重（动态残差插值）
    AIO_WEIGHT = 0.5
    
    # Z 值惩罚参数
    ALPHA_Z = 1.0
    BETA_Z = 5.0
    Z0 = 1.0
    
    # SDF 矩约束
    MU_LO = -0.025
    MU_HI = 0.0
    VAR_HI = 0.25
    
    # ========== 设备配置 ==========
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def firm_state_dim(cls) -> int:
        """返回firm-state向量维度"""
        return 7
    
    @classmethod
    def base_state_dim(cls) -> int:
        """返回base-state向量维度（不含投资成本i）"""
        return 6
    
    @classmethod
    def policy_model_input_var(cls) -> List[int]:
        """返回Policy模型输入的变量索引（不含i）"""
        return [SIMMODEL.B, SIMMODEL.Z, SIMMODEL.ETA, 
                SIMMODEL.X, SIMMODEL.HATCF, SIMMODEL.LNKF]
    
    @classmethod
    def get_ar1_stationary_var(cls, rho: float, sigma: float) -> float:
        """计算AR(1)稳态方差"""
        return sigma**2 / (1 - rho**2)
    
    @classmethod
    def z_stationary_std(cls) -> float:
        """返回z的稳态标准差"""
        return (cls.SIGMA_Z**2 / (1 - cls.RHO_Z**2)) ** 0.5
    
    @classmethod
    def x_stationary_std(cls) -> float:
        """返回x的稳态标准差"""
        return (cls.SIGMA_X**2 / (1 - cls.RHO_X**2)) ** 0.5
