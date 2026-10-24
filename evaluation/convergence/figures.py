"""Figure renderer for the equilibrium convergence evaluator.

Renders the Prompt figure set (F01-F12 main figures plus the V / P / M sub
series) from the result objects produced by the sibling evaluators:

* ``firm_bellman.BellmanEvalResult``  -- ``metrics`` / ``tables`` / ``surfaces``
* ``firm_policy.PolicyEvalResult``    -- ``metrics`` / ``tables`` / ``surfaces``
* ``macro_sdf.MacroSdfEvalResult``    -- ``metrics`` / ``tables`` / ``surfaces``
* ``sim_eval.SimEvalResult``          -- ``metrics`` / ``tables`` / ``surfaces``

Design rules enforced here:

* **No fabricated figures.**  A figure whose data is missing is *skipped*: it
  gets ``status="skipped"``, a stable ``missing_reason`` code, an explanatory
  ``caption`` and **no file on disk**.  A blank "successful" PNG is never
  written.
* **Unified axes and colour scale.**  Every firm surface is drawn on the same
  ``b``/``z`` support and with the same per-family ``(vmin, vmax)`` so that
  surfaces are comparable across episodes and banks.  The applied ranges are
  recorded in ``FigureRecord.meta`` (``b_range`` / ``z_range`` /
  ``value_limits``).
* **CSV / heatmap conventions** of ``evaluation/plotting.py`` are preserved:
  surfaces are indexed ``[b_index, z_index]`` and transposed for
  ``pcolormesh`` / ``plot_surface`` (``x = b``, ``y = z``).
* **Every figure carries a caption** (returned in ``FigureRecord.caption``)
  explaining its economic meaning, and a footer with the Episode label, macro
  anchor, ``eta`` view, ``i`` slice, data distribution source, sample count,
  capital weighting and whether any path went extinct.
* 3D panels use ``mpl_toolkits.mplot3d`` on the Agg backend and always include
  the canonical surface set ``P0 / PI / P / Q_claim / Grid bp0* / Grid bpI*``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)


# ---------------------------------------------------------------------------
# stable reason codes (missing data -> skipped figure)
# ---------------------------------------------------------------------------

REASON_NO_POLICY_SURFACES = "MISSING_POLICY_SURFACES"
REASON_NO_GRID_AXES = "MISSING_BZ_GRID_AXES"
REASON_NO_DEFAULT_BOUNDARY = "MISSING_DEFAULT_BOUNDARY"
REASON_NO_Q_SURFACES = "MISSING_Q_SURFACES"
REASON_NO_BP_SURFACES = "MISSING_BP_SURFACES"
REASON_NO_INVESTMENT_BOUNDARY = "MISSING_INVESTMENT_BOUNDARY"
REASON_NO_BELLMAN_RESULTS = "MISSING_BELLMAN_RESULTS"
REASON_NO_BELLMAN_METRICS = "MISSING_BELLMAN_METRICS"
REASON_NO_POLICY_ROWS = "MISSING_POLICY_ROWS"
REASON_NO_MACRO_PAIRS = "MISSING_MACRO_PAIRS"
REASON_NO_MACRO_ANCHOR_PATHS = "MISSING_MACRO_ANCHOR_PATHS"
REASON_NO_RECURSION_TABLE = "MISSING_FC1_RECURSION_TABLE"
REASON_NO_SDF_TABLE = "MISSING_SDF_TABLE"
REASON_NO_SIM_PANEL = "MISSING_SIM_PANEL"
REASON_NO_JOINT_HIST = "MISSING_JOINT_HISTOGRAM"
REASON_NO_MARGINALS = "MISSING_DISTRIBUTION_MARGINALS"
REASON_NO_CAPITAL_WATERFALL = "MISSING_CAPITAL_WATERFALL"
REASON_NO_SURVIVAL_ACCOUNT = "MISSING_PERIOD_COUNTS"
REASON_NO_OBJECTIVE_SLICES = "MISSING_OBJECTIVE_SLICES"
REASON_INSUFFICIENT_EPISODES = "INSUFFICIENT_EPISODES_FOR_DRIFT"
REASON_POLICY_ROWS_NOT_ALIGNED = "POLICY_ROWS_NOT_ALIGNED_ACROSS_EPISODES"
REASON_INSUFFICIENT_BLOCKS = "INSUFFICIENT_RESULT_BLOCKS"
REASON_NO_FINITE_SAMPLES = "NO_FINITE_SAMPLES"
REASON_NO_MACRO_PANEL = "MISSING_MACRO_PANEL"


# ---------------------------------------------------------------------------
# figure registry
# ---------------------------------------------------------------------------

FIGURE_FILENAMES = {
    "F01": "F01_p0_pi_p_surface.png",
    "F02": "F02_default_boundary_probability.png",
    "F03": "F03_q_caliber_surface_region.png",
    "F04": "F04_grid_bp_investment_boundary.png",
    "F05": "F05_bellman_residual_by_episode.png",
    "F06": "F06_p_q_policy_drift.png",
    "F07": "F07_fc1_pred_vs_actual.png",
    "F08": "F08_macro_dynamics_sdf.png",
    "F09": "F09_leverage_by_macro_state.png",
    "F10": "F10_bz_joint_distribution.png",
    "F11": "F11_capital_waterfall.png",
    "F12": "F12_convergence_dashboard.png",
    "V01": "V01_p0_surface.png",
    "V02": "V02_pi_surface.png",
    "V03": "V03_equity_p_surface.png",
    "V04": "V04_q_four_calibers.png",
    "V05": "V05_value_contour_default_boundary.png",
    "V06": "V06_value_drift_across_episodes.png",
    "P01": "P01_b0_star_surface.png",
    "P02": "P02_bi_star_surface.png",
    "P03": "P03_effective_debt_policy.png",
    "P04": "P04_bellman_objective_vs_bp.png",
    "M01": "M01_fc1_one_step_scatter.png",
    "M02": "M02_hatcf_vs_hatc_path.png",
    "M03": "M03_lnkf_vs_lnk_path.png",
    "M04": "M04_macro_recursive_error.png",
    "M05": "M05_sdf_distribution_constraints.png",
    "M06": "M06_macro_aggregate_time_series.png",
}

FIGURE_IDS = tuple(FIGURE_FILENAMES)

CAPTIONS = {
    "F01": (
        "F01 企业价值与策略曲面（P0/PI/P，3D + 等高线）。P0 是不投资分支的股权价值、"
        "PI 是投资分支的股权价值、P 是投资选择后的混合股权价值；Q_claim 是发行定价用的债权"
        "求偿权价值、Grid bp0*/bpI* 是 Grid-BP 正式策略给出的两个分支最优下一期杠杆。"
        "曲面随生产率 z 递增、随杠杆 b 递减刻画了债务积压（debt overhang）：同样资产净值下"
        "杠杆越高、股权越便宜。等高线图上叠加 Phat=0 违约边界，边界以下为硬违约区。"
    ),
    "F02": (
        "F02 违约边界与违约概率（Gomes 风格 2D）。左列给出软违约概率 bar_z=1-sigmoid(Phat/tau_z)"
        "的曲面与 Phat=0 硬违约边界；bar_z 是软概率、不是生产率门槛。右列给出硬边界 z_default(b)、"
        "软边界 bar_z=0.5 与条件违约概率的对照。经济含义：违约边界随杠杆 b 上移（越容易违约），"
        "硬/软边界之差度量软惩罚与硬门之间的口径落差。"
    ),
    "F03": (
        "F03 Q 四口径曲面与违约区（Q Claim / Q Unit / Recovery / Q Effective）。Q_claim=b₊·q_unit"
        "是发行定价与 Bellman 残差唯一允许的口径；q_unit 是单位债券价；recovery=φ(1-δ+exp(x+z)) 是"
        "违约回收值；Q_effective 是结算口径（hybrid 才含违约硬门）。图中深色区域为 Phat≤0 的硬违约区。"
        "经济含义：违约区求偿权价值塌缩到回收值，存活区 Q_claim 随 b 先升后降体现最优资本结构。"
    ),
    "F04": (
        "F04 Grid BP 策略与投资边界（3D + 2D）。Grid bp0*/bpI* 是 Grid-BP 教师在候选杠杆网格上"
        "求得的正式策略（PolicyValueModel 的 bp 头仅兼容输出），投资边界 i*(b,z) 是 PI-V0 沿投资成本"
        "轴的零点。经济含义：生产率越高、最优杠杆越高；投资边界以下才投资，且多零点/无零点状态被"
        "显式分类（always/never/single/multiple）。"
    ),
    "F05": (
        "F05 Bellman 残差随 Episode 收敛（Line + P90）。frozen 残差是训练冻结策略下的 Bellman 恒等式"
        "残差，fresh 残差是当前模型重新贪心求解后的最优性残差（最优性证据以 fresh 为准）。"
        "P90/P99 与 MAE 同画以避免均值掩盖尾部。经济含义：残差随 episode 下降到量级 1e-3 以内才构成"
        "均衡收敛的必要条件；训练 loss 不作收敛证据。"
    ),
    "F06": (
        "F06 P / Q / Policy 跨 Episode 漂移（多面板 Line）。在同一固定 EvalBank 状态上逐行比较相邻"
        "episode 的 P（股权价值）、Q（发行定价）与 Grid BP 策略，另报 policy/default/investment flip rate。"
        "经济含义：函数漂移趋零是均衡收敛的直接证据；flip rate 刻画决策面的离散翻转比例。"
    ),
    "F07": (
        "F07 FC1 预测 vs 实际（Scatter + Path）。hatcf_t/lnkf_t 是 t-1 对节点 t 的信念预测，"
        "hatc_t/lnk_t 是节点 t 的实现聚合，同行对齐即 shift=0 的一步预测。经济含义：预测点越贴近 45° 线"
        "信念体系越准；偏离 45° 的系统性斜率/截距反映预测偏差而非噪声。"
    ),
    "F08": (
        "F08 宏观动态与 SDF（Time Series）。给出 Bank C 宏观锚点（Recession/Normal/Expansion）上"
        "FC1 递归路径、SDF 条件残差按宏观态分解与 M 的均值/标准误。经济含义：递归路径是否漂移出训练范围"
        "决定长期模拟可信度；SDF 条件矩在三态间应稳定，否则资产定价核不一致。"
    ),
    "F09": (
        "F09 分宏观态杠杆分布（Gomes 风格直方图）。按 x 分位把截面切成 Recession/Normal/Expansion，"
        "分别给出企业计数加权与资本加权两种口径的杠杆分布。经济含义：衰退态分布应左移（去杠杆）、"
        "资本加权口径右移（大企业杠杆更高）；两种口径的差异即规模-杠杆相关性。"
    ),
    "F10": (
        "F10 b-z 联合分布随时间（Heatmap + 边缘分布）。固定网格上的归一化联合直方图与其 b/z 边缘分布，"
        "计数加权与资本加权双口径。经济含义：联合分布随时间的形态演化反映选择效应与内生退出；"
        "灭绝 path 显式标注、不从统计中删除。"
    ),
    "F11": (
        "F11 t0→t1 资本 Waterfall 与存活核算。按资本增长分解恒等式逐期画瀑布：期末资本 = 期初资本 "
        "+ 存续企业 ΔK + 进入资本（端点口径）- 退出资本 + 记账残差；同节点进出企业是备忘行、不进恒等式。"
        "并排给出存活/违约/活跃 path 逐期账。经济含义：残差应贴近 0，否则资本账本口径有缺口。"
    ),
    "F12": (
        "F12 一般均衡收敛 Dashboard。汇总 Bellman 残差、P/Q/策略漂移、FC1 一步误差、SDF 矩与分布距离"
        "随 episode 的轨迹。经济含义：只有当企业价值/策略、宏观信念与定价核同时稳定时，才是一般均衡意义"
        "上的收敛；任一面板不收敛即整体不收敛。"
    ),
    "V01": (
        "V01 P0 曲面（不投资分支股权价值）。叠加 Phat=0 违约边界；η=0 与 η=1 两个反事实视图并列，"
        "统一 b/z 范围与色标以便跨集比较。经济含义：P0 是放弃投资机会后的清算-存续价值。"
    ),
    "V02": (
        "V02 PI 曲面（投资分支股权价值，标注投资成本 i）。在固定参考投资成本 i_mid（10/50/90 分位的"
        "中位档）处求值。经济含义：PI-P0 的符号决定是否投资，i 越高投资分支越不利。"
    ),
    "V03": (
        "V03 股权价值 P 与 Phat≤0 违约区。P 是投资选择后的混合股权价值，Phat≤0 是硬违约指示（模型口径），"
        "与 bar_z 软概率严格区分。经济含义：违约区内股权价值归零/塌缩，边界附近的价值梯度最大。"
    ),
    "V04": (
        "V04 Q 四口径对照（Q Claim / Q Unit / Q Effective / Recovery）。发行与残差一律使用 claim 口径，"
        "effective 与回收值只用于结算语义。经济含义：四口径在违约区的分离度量了违约硬门与软惩罚的差异。"
    ),
    "V05": (
        "V05 价值等高线 + 违约边界。P0/PI/P 的等高线叠加 Phat=0 硬边界与 bar_z=0.5 软边界。"
        "经济含义：等高线在违约边界处的折点显示债务积压对边际价值的抑制。"
    ),
    "V06": (
        "V06 跨集价值漂移（Value Drift）。相邻 episode 之间 P0/PI/P 曲面的逐点差与汇总统计。"
        "经济含义：漂移趋零说明值函数在固定 bank 状态上已稳定，是收敛的必要证据。"
    ),
    "P01": (
        "P01 Grid bp0* 曲面（不投资分支最优下一期杠杆）。来自 Grid-BP 教师的贪心解，不是 bp0 头输出。"
        "经济含义：最优杠杆随生产率上升、随当前杠杆下降（均值回复）。"
    ),
    "P02": (
        "P02 Grid bpI* 曲面（投资分支最优下一期杠杆）。投资分支需要额外融资，最优杠杆系统性高于 bp0*。"
    ),
    "P03": (
        "P03 有效债务策略（实际 b_next）。b_next = η·bp* + (1-η)·b：η=0 时强制 b_next=b（再融资关闭），"
        "η=1 时 b_next=bp*。图中同时给出头输出 vs Grid BP 的差距。经济含义：融资门决定了策略的实际执行，"
        "意愿策略与执行策略必须分开统计。"
    ),
    "P04": (
        "P04 Bellman 目标 vs 候选 bp（叠加 Q Claim）。代表性状态下 Bellman 右端随候选杠杆的变化与"
        "发行 Q_claim 的对照。经济含义：目标曲线的峰值即 Grid bp*，η=0 行必须是常数（融资门强制）。"
    ),
    "M01": (
        "M01 FC1 一步预测散点。hatcf_t vs hatc_t、lnkf_t vs lnk_t（shift=0 同行对齐），附 45° 线与 R²。"
        "经济含义：预测无偏且贴线是宏观信念体系一致的必要条件。"
    ),
    "M02": (
        "M02 Hatcf vs Hatc 路径。按 path 画预测消费增长率信念与实现值的时间路径。"
        "经济含义：信念应围绕实现值波动而非系统性偏离。"
    ),
    "M03": (
        "M03 LnKF vs LnK 路径。按 path 画资本信念与实现值的时间路径。"
        "经济含义：资本信念的持续偏离会累积成错误的融资/投资决策。"
    ),
    "M04": (
        "M04 宏观递归预测误差。production_one_step（每次喂父节点算得宏观态，正式模拟调用规则）与 "
        "autonomous_recursive（完全自回归）两口径分开报，误差随 horizon 的斜率即漂移速度。"
        "经济含义：autonomous 口径的漂移斜率决定长期模拟的可信上限。"
    ),
    "M05": (
        "M05 SDF 分布与约束。M 的均值/方差、按宏观态分解的条件残差、Signed AiO 估计（含标准误与解读护栏）"
        "以及矩约束违反量 g_mean_low/g_mean_high/g_var_high。经济含义：负的 Signed AiO 样本均值不等于负的"
        "总体条件矩；约束违反量才是数值安全的判据。"
    ),
    "M06": (
        "M06 C/K、K、C、Y、I 随时间。宏观聚合量的逐期轨迹（C/K 由 Hatc=log(C/K) 还原）。"
        "经济含义：宏观稳态意味着这些聚合量不再漂移；持续趋势说明一般均衡未收敛。"
    ),
}

# Canonical 3D surface set required on every 3D figure.
CANONICAL_3D_NAMES = ("P0", "PI", "P", "Q_claim", "Grid bp0*", "Grid bpI*")


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FigureStyle:
    """Unified paper-grade style plus the shared axis / colour scale."""

    dpi: int = 160
    font_size: float = 9.0
    title_size: float = 10.5
    label_size: float = 8.5
    legend_size: float = 7.5
    footer_size: float = 6.8
    cmap_surface: str = "viridis"
    cmap_residual: str = "coolwarm"
    cmap_density: str = "viridis"
    cmap_region: str = "Reds"
    contour_levels: int = 20
    elev: float = 28.0
    azim: float = -132.0
    b_range: Optional[Tuple[float, float]] = None
    z_range: Optional[Tuple[float, float]] = None
    value_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def limit_for(self, family, values):
        limits = self.value_limits.get(family)
        if limits is not None:
            return float(limits[0]), float(limits[1])
        array = _finite(values)
        if array.size == 0:
            return None, None
        return float(np.min(array)), float(np.max(array))


@dataclass
class FigureRecord:
    """One rendered (or deliberately skipped) figure."""

    figure_id: str
    path: Optional[str] = None
    caption: str = ""
    status: str = "skipped"
    missing_reason: Optional[str] = None
    n_samples: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "figure_id": self.figure_id,
            "path": self.path,
            "caption": self.caption,
            "status": self.status,
            "missing_reason": self.missing_reason,
            "n_samples": self.n_samples,
            "meta": dict(self.meta),
        }


@dataclass
class FiguresManifest:
    """Result of :func:`render_all_figures`."""

    figures: Dict[str, FigureRecord] = field(default_factory=dict)
    output_dir: Optional[str] = None
    style: Optional[FigureStyle] = None

    def __iter__(self):
        return iter(self.figures.values())

    def __len__(self) -> int:
        return len(self.figures)

    def __getitem__(self, figure_id: str) -> FigureRecord:
        return self.figures[figure_id]

    @property
    def ok(self):
        return {key: value for key, value in self.figures.items() if value.status == "ok"}

    @property
    def skipped(self):
        return {key: value for key, value in self.figures.items() if value.status != "ok"}

    def to_dict(self):
        return {
            "output_dir": self.output_dir,
            "figures": {key: value.to_dict() for key, value in self.figures.items()},
        }


@dataclass
class _Panel:
    """Declarative panel description consumed by :func:`_render_grid`."""

    kind: str
    title: str = ""
    family: str = "value"
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""
    values: Any = None
    options: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------


def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _n_finite(values) -> int:
    return int(_finite(values).size)


def _episode_key(value) -> Tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _as_episode_map(value) -> Dict[Any, Any]:
    """Normalize ``{episode: result}`` / ``[result]`` / single result inputs."""
    if value is None:
        return {}
    items: List[Tuple[Any, Any]] = []
    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            episode = None
            diagnostics = getattr(item, "diagnostics", None)
            if isinstance(diagnostics, Mapping):
                episode = diagnostics.get("episode")
            items.append((episode if episode is not None else index, item))
    else:
        diagnostics = getattr(value, "diagnostics", None)
        episode = diagnostics.get("episode") if isinstance(diagnostics, Mapping) else None
        items = [(episode if episode is not None else 0, value)]
    ordered = sorted(items, key=lambda kv: _episode_key(kv[0]))
    return {key: item for key, item in ordered}


def _result_tables(result):
    tables = getattr(result, "tables", None)
    return dict(tables) if isinstance(tables, Mapping) else {}


def _result_surfaces(result):
    surfaces = getattr(result, "surfaces", None)
    return dict(surfaces) if isinstance(surfaces, Mapping) else {}


def _result_metrics(result):
    metrics = getattr(result, "metrics", None)
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def _pick_table(result, base: str) -> Optional[pd.DataFrame]:
    """``base`` first, then any ``base_<slug>`` (sim_eval variant suffixes)."""
    tables = _result_tables(result)
    frame = tables.get(base)
    if isinstance(frame, pd.DataFrame):
        return frame
    suffixes = sorted(key for key in tables if str(key).startswith(base + "_"))
    for key in suffixes:
        candidate = tables[key]
        if isinstance(candidate, pd.DataFrame) and not candidate.empty:
            return candidate
    return None


def _table_column(frame, name: str):
    if frame is None or name not in frame.columns:
        return None
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)


_BZ_HIST_KEY = re.compile(r"^bz_hist_t(\d+)_(count|capital)(?:_(.+))?$")


def _joint_hists(result):
    out = {}
    by_key = {}
    for key, array in _result_surfaces(result).items():
        match = _BZ_HIST_KEY.match(str(key))
        if match is None:
            continue
        slot = (int(match.group(1)), str(match.group(2)))
        by_key.setdefault(slot, {})[str(match.group(3) or "")] = np.asarray(array)
    for slot, variants in by_key.items():
        if "" in variants:
            out[slot] = variants[""]
        else:
            out[slot] = variants[sorted(variants)[0]]
    return out


def _cmap(name: str):
    try:
        return matplotlib.colormaps[name]
    except Exception:  # pragma: no cover - older matplotlib
        return plt.get_cmap(name)


# ---------------------------------------------------------------------------
# style resolution (unified b/z support and colour limits)
# ---------------------------------------------------------------------------

_FAMILY_LIMIT_KEYS = (
    ("value", ("P0", "PI", "P", "V0", "VI", "value_star", "bellman_rhs", "continuation")),
    ("q", ("Q_claim", "Q_effective", "Q", "q_unit", "recovery", "q_claim_issue")),
    ("bp", ("bp0", "bpI", "bp", "b_next", "bp_candidate", "i_star")),
    ("prob", ("bar_z", "bar_i", "survival_prob", "default", "share", "rate", "prob")),
    ("residual", ("R0", "RI", "RQ", "residual", "error", "drift", "diff", "gap")),
)


def _family_of(name: str) -> str:
    text = str(name)
    for family, tokens in _FAMILY_LIMIT_KEYS:
        for token in tokens:
            if token in text:
                return family
    return "value"


def _collect_ranges(results):
    b_values = []
    z_values = []
    for result in results:
        for name in ("investment_boundary", "default_boundary", "hard_soft_boundary"):
            frame = _pick_table(result, name)
            if frame is None:
                continue
            for column, sink in (("b", b_values), ("z", z_values)):
                values = _table_column(frame, column)
                if values is not None:
                    good = _finite(values)
                    if good.size:
                        sink.append(float(np.min(good)))
                        sink.append(float(np.max(good)))
    b_range = (min(b_values), max(b_values)) if b_values else None
    z_range = (min(z_values), max(z_values)) if z_values else None
    return b_range, z_range


def _collect_value_limits(results):
    collected = {}
    for result in results:
        for name, values in _result_surfaces(result).items():
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 2:
                continue
            good = _finite(array)
            if good.size == 0:
                continue
            bucket = collected.setdefault(_family_of(name), [])
            bucket.append(float(np.min(good)))
            bucket.append(float(np.max(good)))
    return {family: (min(values), max(values)) for family, values in collected.items()}


def resolve_style(style: Optional[FigureStyle], results) -> FigureStyle:
    """Fill the shared ``b``/``z`` support and colour limits on ``style``.

    Ranges are the global extent over the supplied fixed-bank results, so every
    surface of every episode is drawn on one identical coordinate box and one
    identical colour scale.
    """
    base = replace(style) if style is not None else FigureStyle()
    if base.b_range is None or base.z_range is None:
        b_range, z_range = _collect_ranges(results)
        if base.b_range is None:
            base.b_range = b_range
        if base.z_range is None:
            base.z_range = z_range
    if not base.value_limits:
        base.value_limits = _collect_value_limits(results)
    return base


def _style_meta(style: FigureStyle):
    return {
        "b_range": list(style.b_range) if style.b_range is not None else None,
        "z_range": list(style.z_range) if style.z_range is not None else None,
        "value_limits": {
            family: [float(low), float(high)]
            for family, (low, high) in sorted(style.value_limits.items())
        },
        "cmap_surface": style.cmap_surface,
        "contour_levels": int(style.contour_levels),
        "unified_scale": True,
    }


# ---------------------------------------------------------------------------
# annotations / footer
# ---------------------------------------------------------------------------


def _annotation_lines(annotations, **overrides):
    values = {
        "Episode": "未提供",
        "Macro Anchor": "未提供",
        "η": "未提供",
        "i": "未提供",
        "数据分布来源": "未提供",
        "n": "未提供",
        "资本加权": "未提供",
        "灭绝 path": "未提供",
    }
    if isinstance(annotations, Mapping):
        values.update({str(key): value for key, value in annotations.items()})
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return [f"{key}={value}" for key, value in values.items()]


def _describe_macro_anchor(bank) -> str:
    if not isinstance(bank, Mapping):
        return "未提供"
    candidates = [bank]
    for key in ("C", "c", "bank_c", "bank_C"):
        if isinstance(bank.get(key), Mapping):
            candidates.append(bank[key])
    for candidate in candidates:
        anchor = candidate.get("macro_anchor")
        if isinstance(anchor, Mapping) and anchor:
            labels = sorted(str(key) for key in anchor)
            first = next(iter(anchor.values()))
            quantile = first.get("quantile") if isinstance(first, Mapping) else None
            suffix = f" @x q={quantile}" if quantile is not None else ""
            return f"{'/'.join(labels)}{suffix}"
        metadata = candidate.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("macro_anchor_source"):
            return str(metadata["macro_anchor_source"])
    return "未提供"


def _describe_bank_source(bank) -> str:
    if not isinstance(bank, Mapping):
        return "未提供"
    bank_id = bank.get("bank_id")
    bank_kind = bank.get("bank_kind")
    if bank_id or bank_kind:
        return f"{bank_id or 'bank'} ({bank_kind or 'unknown'})"
    for key in ("A", "B", "C", "D", "E"):
        entry = bank.get(key)
        if isinstance(entry, Mapping) and entry.get("bank_id"):
            return f"{entry['bank_id']} ({entry.get('bank_kind', 'unknown')})"
    return "未提供"


def _describe_extinct(sim_result) -> str:
    if isinstance(sim_result, Mapping):
        episodes = _as_episode_map(sim_result)
        if not episodes:
            return "未提供"
        sim_result = list(episodes.values())[-1]
    status = _pick_table(sim_result, "path_status")
    if status is None or "extinct" not in status.columns:
        return "未提供"
    flags = status["extinct"].astype(bool).to_numpy()
    return f"{int(flags.sum())}/{int(flags.size)}"


# ---------------------------------------------------------------------------
# drawing core
# ---------------------------------------------------------------------------


def _apply_style(style: FigureStyle) -> None:
    plt.rcParams.update(
        {
            "font.size": style.font_size,
            "axes.titlesize": style.title_size,
            "axes.labelsize": style.label_size,
            "legend.fontsize": style.legend_size,
            "xtick.labelsize": style.label_size,
            "ytick.labelsize": style.label_size,
            "figure.dpi": style.dpi,
            "savefig.dpi": style.dpi,
            "axes.grid": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
        }
    )


def _save(fig, path: Path, style: FigureStyle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=style.dpi, metadata={"Software": "convergence-eval-figures"})
    plt.close(fig)


def _set_xy_limits(ax, style: FigureStyle) -> None:
    if style.b_range is not None:
        ax.set_xlim(float(style.b_range[0]), float(style.b_range[1]))
    if style.z_range is not None:
        ax.set_ylim(float(style.z_range[0]), float(style.z_range[1]))


def _draw_surface_3d(ax, panel: _Panel, style: FigureStyle) -> bool:
    b_values = np.asarray(panel.options["b_values"], dtype=np.float64)
    z_values = np.asarray(panel.options["z_values"], dtype=np.float64)
    values = np.asarray(panel.values, dtype=np.float64)
    if values.shape != (b_values.size, z_values.size):
        return False
    mesh_b, mesh_z = np.meshgrid(b_values, z_values, indexing="ij")
    low, high = style.limit_for(panel.family, values)
    norm = None
    if low is not None and high is not None and math.isfinite(low) and math.isfinite(high):
        if high <= low:
            high = low + 1e-9
        norm = matplotlib.colors.Normalize(vmin=low, vmax=high)
    ax.plot_surface(
        mesh_b,
        mesh_z,
        values,
        cmap=_cmap(style.cmap_residual if panel.family == "residual" else style.cmap_surface),
        norm=norm,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )
    ax.set_xlabel(panel.xlabel or "b = 杠杆率 (债务/资本)")
    ax.set_ylabel(panel.ylabel or "z = 企业生产率冲击")
    ax.set_zlabel(panel.zlabel or panel.title)
    ax.view_init(elev=style.elev, azim=style.azim)
    ax.set_title(panel.title)
    if style.b_range is not None:
        ax.set_xlim3d(float(style.b_range[0]), float(style.b_range[1]))
    if style.z_range is not None:
        ax.set_ylim3d(float(style.z_range[0]), float(style.z_range[1]))
    return True


def _overlay_lines(ax, panel: _Panel) -> None:
    overlays = panel.options.get("overlays") or []
    for item in overlays:
        x = np.asarray(item.get("x"), dtype=np.float64)
        y = np.asarray(item.get("y"), dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            continue
        ax.plot(
            x[mask],
            y[mask],
            color=item.get("color", "black"),
            linestyle=item.get("linestyle", "-"),
            linewidth=item.get("linewidth", 1.4),
            label=item.get("label"),
        )
    if overlays:
        ax.legend()


def _shade_regions(ax, panel: _Panel, style: FigureStyle) -> None:
    region = panel.options.get("region")
    if region is None:
        return
    mask = np.asarray(region, dtype=bool)
    if mask.shape != np.asarray(panel.values).shape:
        return
    if not bool(mask.any()):
        return
    ax.contourf(
        np.asarray(panel.options["b_values"], dtype=np.float64),
        np.asarray(panel.options["z_values"], dtype=np.float64),
        mask.astype(np.float64).T,
        levels=[0.5, 1.5],
        colors=["#b2182b"],
        alpha=0.32,
    )


def _draw_contour(ax, panel: _Panel, style: FigureStyle) -> bool:
    b_values = np.asarray(panel.options["b_values"], dtype=np.float64)
    z_values = np.asarray(panel.options["z_values"], dtype=np.float64)
    values = np.asarray(panel.values, dtype=np.float64)
    if values.shape != (b_values.size, z_values.size):
        return False
    low, high = style.limit_for(panel.family, values)
    if low is None or not math.isfinite(low) or not math.isfinite(high) or high <= low:
        good = _finite(values)
        if good.size == 0:
            return False
        low, high = float(np.min(good)), float(np.max(good))
        if high <= low:
            high = low + 1e-9
    levels = np.linspace(low, high, int(style.contour_levels) + 1)
    mesh = ax.contourf(
        b_values,
        z_values,
        values.T,
        levels=levels,
        cmap=_cmap(style.cmap_residual if panel.family == "residual" else style.cmap_surface),
    )
    _shade_regions(ax, panel, style)
    _overlay_lines(ax, panel)
    fig = ax.get_figure()
    if fig is not None:
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel(panel.xlabel or "b = 杠杆率 (债务/资本)")
    ax.set_ylabel(panel.ylabel or "z = 企业生产率冲击")
    ax.set_title(panel.title)
    _set_xy_limits(ax, style)
    return True


def _draw_heatmap(ax, panel: _Panel, style: FigureStyle) -> bool:
    b_values = np.asarray(panel.options["b_values"], dtype=np.float64)
    z_values = np.asarray(panel.options["z_values"], dtype=np.float64)
    values = np.asarray(panel.values, dtype=np.float64)
    if values.shape != (b_values.size, z_values.size):
        return False
    low, high = style.limit_for(panel.family, values)
    mesh = ax.pcolormesh(
        b_values,
        z_values,
        values.T,
        shading="auto",
        cmap=_cmap(style.cmap_residual if panel.family == "residual" else style.cmap_surface),
        vmin=low,
        vmax=high,
    )
    _shade_regions(ax, panel, style)
    _overlay_lines(ax, panel)
    fig = ax.get_figure()
    if fig is not None:
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel(panel.xlabel or "b = 杠杆率 (债务/资本)")
    ax.set_ylabel(panel.ylabel or "z = 企业生产率冲击")
    ax.set_title(panel.title)
    _set_xy_limits(ax, style)
    return True


def _draw_line(ax, panel: _Panel, style: FigureStyle) -> bool:
    drawn = False
    for series in panel.options.get("series", []) or []:
        x = np.asarray(series.get("x"), dtype=np.float64)
        y = np.asarray(series.get("y"), dtype=np.float64)
        if x.size == 0:
            continue
        ax.plot(
            x,
            y,
            marker=series.get("marker", "o"),
            markersize=series.get("markersize", 3.5),
            linewidth=series.get("linewidth", 1.3),
            linestyle=series.get("linestyle", "-"),
            color=series.get("color"),
            label=series.get("label"),
        )
        drawn = True
    if panel.options.get("logy") == "symlog":
        ax.set_yscale("symlog")
    elif panel.options.get("logy"):
        ax.set_yscale("log")
    ax.set_xlabel(panel.xlabel)
    ax.set_ylabel(panel.ylabel)
    ax.set_title(panel.title)
    ax.grid(alpha=0.25)
    if panel.options.get("series"):
        ax.legend()
    return drawn


def _draw_scatter(ax, panel: _Panel, style: FigureStyle) -> bool:
    drawn = False
    for series in panel.options.get("series", []) or []:
        x = np.asarray(series.get("x"), dtype=np.float64)
        y = np.asarray(series.get("y"), dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            continue
        ax.scatter(
            x[mask],
            y[mask],
            s=series.get("size", 8),
            alpha=series.get("alpha", 0.55),
            label=series.get("label"),
            color=series.get("color"),
            edgecolors="none",
        )
        drawn = True
    if panel.options.get("diag", True):
        all_values = []
        for series in panel.options.get("series", []) or []:
            all_values.append(np.asarray(series.get("x"), dtype=np.float64))
            all_values.append(np.asarray(series.get("y"), dtype=np.float64))
        joined = _finite(np.concatenate(all_values)) if all_values else np.empty(0)
        if joined.size:
            low, high = float(np.min(joined)), float(np.max(joined))
            ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(panel.xlabel)
    ax.set_ylabel(panel.ylabel)
    ax.set_title(panel.title)
    ax.grid(alpha=0.25)
    if panel.options.get("series"):
        ax.legend()
    return drawn


def _draw_bar(ax, panel: _Panel, style: FigureStyle) -> bool:
    labels = list(panel.options.get("labels", []) or [])
    values = np.asarray(panel.options.get("values", []), dtype=np.float64)
    if labels and values.size:
        positions = np.arange(len(labels))
        ax.bar(
            positions,
            values,
            yerr=panel.options.get("yerr"),
            color=panel.options.get("color", "tab:blue"),
            alpha=0.85,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=panel.options.get("rotation", 0))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(panel.xlabel)
        ax.set_ylabel(panel.ylabel)
        ax.set_title(panel.title)
        ax.grid(alpha=0.2, axis="y")
        return True
    return False


def _draw_text(ax, panel: _Panel, style: FigureStyle) -> bool:
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        str(panel.options.get("text", "")),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=style.legend_size,
    )
    ax.set_title(panel.title)
    return True


def _draw_hist(ax, panel: _Panel, style: FigureStyle) -> bool:
    drawn = False
    for series in panel.options.get("series", []) or []:
        centers = np.asarray(series.get("x"), dtype=np.float64)
        values = np.asarray(series.get("y"), dtype=np.float64)
        if centers.size == 0:
            continue
        ax.plot(
            centers,
            values,
            drawstyle=series.get("drawstyle", "steps-mid"),
            linewidth=series.get("linewidth", 1.3),
            label=series.get("label"),
            color=series.get("color"),
        )
        if panel.options.get("fill", False):
            ax.fill_between(centers, values, step="mid", alpha=0.18)
        drawn = True
    ax.set_xlabel(panel.xlabel)
    ax.set_ylabel(panel.ylabel)
    ax.set_title(panel.title)
    ax.grid(alpha=0.25)
    if panel.options.get("series"):
        ax.legend()
    return drawn


_DRAW = {
    "3d": _draw_surface_3d,
    "contour": _draw_contour,
    "heatmap": _draw_heatmap,
    "line": _draw_line,
    "scatter": _draw_scatter,
    "bar": _draw_bar,
    "text": _draw_text,
    "hist": _draw_hist,
}


def _render_grid(
    figure_id: str,
    panels: Sequence[_Panel],
    *,
    rows: int,
    cols: int,
    output_dir: Path,
    style: FigureStyle,
    caption: str,
    meta: Dict[str, Any],
    n_samples,
    annotations=None,
    figsize=None,
) -> FigureRecord:
    """Draw ``rows x cols`` panels; skip the figure when nothing is drawable."""
    filename = FIGURE_FILENAMES[figure_id]
    output_dir = Path(output_dir)
    full_meta = dict(meta)
    full_meta.update(_style_meta(style))
    full_meta["panels"] = [panel.title for panel in panels]
    full_meta["dropped_panels"] = [panel.title for panel in panels[rows * cols :]]
    full_meta["annotations"] = _annotation_lines(annotations)

    drawable = [
        panel
        for panel in panels
        if panel.values is not None or panel.kind in ("text", "line", "scatter", "bar", "hist")
    ]
    if not drawable:
        return FigureRecord(
            figure_id=figure_id,
            caption=caption,
            status="skipped",
            missing_reason=str(meta.get("missing_reason") or REASON_NO_FINITE_SAMPLES),
            n_samples=n_samples,
            meta=full_meta,
        )

    _apply_style(style)
    width = float(figsize[0]) if figsize else 5.0 * cols
    height = float(figsize[1]) if figsize else 4.0 * rows
    fig = plt.figure(figsize=(width, height))
    drawn_any = False
    used = []
    for index, panel in enumerate(panels[: rows * cols]):
        projection = "3d" if panel.kind == "3d" else None
        ax = fig.add_subplot(rows, cols, index + 1, projection=projection)
        try:
            drawn = bool(_DRAW[panel.kind](ax, panel, style))
        except Exception as exc:  # a malformed panel must not fake a success figure
            drawn = False
            panel.title = f"{panel.title} [绘制失败: {type(exc).__name__}: {str(exc)[:160]}]"
        drawn_any = drawn_any or drawn
        used.append({"title": panel.title, "drawn": bool(drawn)})
    full_meta["panel_status"] = used
    full_meta["panel_errors"] = [
        item["title"] for item in used if not item["drawn"]
    ]

    if not drawn_any:
        plt.close(fig)
        return FigureRecord(
            figure_id=figure_id,
            caption=caption,
            status="skipped",
            missing_reason=str(meta.get("missing_reason") or REASON_NO_FINITE_SAMPLES),
            n_samples=n_samples,
            meta=full_meta,
        )

    fig.suptitle(f"{figure_id}: {caption.split('。')[0]}", fontsize=style.title_size + 1.5)
    fig.subplots_adjust(
        left=0.07, right=0.955, top=0.90, bottom=0.115, hspace=0.55, wspace=0.32
    )
    fig.text(
        0.005,
        0.005,
        " | ".join(_annotation_lines(annotations)),
        ha="left",
        va="bottom",
        fontsize=style.footer_size,
    )
    path = output_dir / filename
    _save(fig, path, style)
    return FigureRecord(
        figure_id=figure_id,
        path=str(path),
        caption=caption,
        status="ok",
        missing_reason=None,
        n_samples=n_samples,
        meta=full_meta,
    )


def _skip(figure_id: str, reason: str, meta=None) -> FigureRecord:
    return FigureRecord(
        figure_id=figure_id,
        path=None,
        caption=CAPTIONS.get(figure_id, ""),
        status="skipped",
        missing_reason=reason,
        n_samples=None,
        meta=dict(meta or {}),
    )


# ---------------------------------------------------------------------------
# firm-surface panel builders
# ---------------------------------------------------------------------------

_GRID_SOURCE_TABLES = ("investment_boundary", "default_boundary", "hard_soft_boundary")


def grid_axes(result):
    """Recover the ``(b_values, z_values)`` surface axes of a policy result.

    The surfaces themselves are ``(len(b), len(z))`` arrays; the axes come from
    the boundary tables, which carry the grid coordinates explicitly.
    """
    for name in _GRID_SOURCE_TABLES:
        frame = _pick_table(result, name)
        if frame is None:
            continue
        if {"b", "z"}.issubset(frame.columns):
            b_values = np.unique(_finite(_table_column(frame, "b")))
            z_values = np.unique(_finite(_table_column(frame, "z")))
            if b_values.size >= 2 and z_values.size >= 2:
                return b_values, z_values
        if {"b", "z_default"}.issubset(frame.columns):
            b_values = np.unique(_finite(_table_column(frame, "b")))
            z_values = np.unique(_finite(_table_column(frame, "z_default")))
            if b_values.size >= 2 and z_values.size >= 2:
                return b_values, z_values
    return None


def _surface(result, name: str):
    values = _result_surfaces(result).get(name)
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    return array if array.ndim == 2 else None


def _canonical_surfaces(result):
    """The canonical 3D set; ``None`` + reason when any member is missing."""
    mapping = {
        "P0": "eta1_P0",
        "PI": "eta1_PI",
        "P": "eta1_P",
        "Q_claim": "eta1_Q_claim",
        "Grid bp0*": "eta1_bp0",
        "Grid bpI*": "eta1_bpI",
    }
    out = {}
    for label, key in mapping.items():
        values = _surface(result, key)
        if values is None:
            return None, REASON_NO_POLICY_SURFACES
        out[label] = values
    return out, None


def _boundary_overlay(result, style: FigureStyle):
    overlays = []
    frame = _pick_table(result, "default_boundary")
    if frame is not None and {"b", "z_default"}.issubset(frame.columns):
        b = _table_column(frame, "b")
        z = _table_column(frame, "z_default")
        if b is not None and z is not None:
            overlays.append({"x": b, "y": z, "label": "违约边界 Phat=0", "color": "black"})
    hard = _pick_table(result, "hard_soft_boundary")
    if hard is not None and {"b", "z_soft_barz0p5"}.issubset(hard.columns):
        b = _table_column(hard, "b")
        z = _table_column(hard, "z_soft_barz0p5")
        if b is not None and z is not None:
            overlays.append(
                {
                    "x": b,
                    "y": z,
                    "label": "软边界 bar_z=0.5",
                    "color": "tab:red",
                    "linestyle": "--",
                }
            )
    return overlays


def _or_nan(values, shape):
    return values if values is not None else np.full(shape, np.nan)


def _default_region(result, grid_shape):
    region = _surface(result, "eta1_default_region")
    if region is not None and region.shape == grid_shape:
        return region.astype(bool)
    phat = _surface(result, "eta1_Phat")
    if phat is not None and phat.shape == grid_shape:
        return phat <= 0.0
    return None


def _eta_label(view: str) -> str:
    return "η=1（refi_active，b_next=bp*）" if view == "eta1" else "η=0（refi 关闭，b_next=b）"


def _base_meta(
    *,
    episode=None,
    eta=None,
    i_slice=None,
    data_source=None,
    capital_weighted=None,
    extinct_paths=None,
    n_samples=None,
    extra=None,
):
    meta = {
        "episode": episode,
        "eta_view": eta,
        "i_slice": i_slice,
        "data_source": data_source,
        "capital_weighted": capital_weighted,
        "extinct_paths": extinct_paths,
        "n_samples": n_samples,
    }
    if extra:
        meta.update(extra)
    return meta


def _annotation_dict(
    annotations,
    *,
    episode=None,
    eta=None,
    i_slice=None,
    data_source=None,
    capital_weighted=None,
    extinct_paths=None,
    n_samples=None,
):
    merged = dict(annotations or {})
    if episode is not None:
        merged["Episode"] = episode
    if eta is not None:
        merged["η"] = eta
    if i_slice is not None:
        merged["i"] = i_slice
    if data_source is not None:
        merged["数据分布来源"] = data_source
    if capital_weighted is not None:
        merged["资本加权"] = capital_weighted
    if extinct_paths is not None:
        merged["灭绝 path"] = extinct_paths
    if n_samples is not None:
        merged["n"] = n_samples
    return merged


def _select_episode(policy_results, episode):
    episodes = _as_episode_map(policy_results)
    if not episodes:
        return None, None, REASON_NO_POLICY_SURFACES
    if episode is None:
        key = list(episodes)[-1]
    elif episode in episodes:
        key = episode
    else:
        target = _episode_key(episode)
        match = [key for key in episodes if _episode_key(key) == target]
        if not match:
            return None, None, REASON_NO_POLICY_SURFACES
        key = match[0]
    return key, episodes[key], None


def _i_slice_text(policy_result) -> str:
    diagnostics = getattr(policy_result, "diagnostics", None) or {}
    reference = diagnostics.get("reference_i") if isinstance(diagnostics, Mapping) else None
    if reference is not None:
        return f"i_mid={reference}"
    return "i=i_mid（bank i 中位数）"


# ---------------------------------------------------------------------------
# F-series renderers
# ---------------------------------------------------------------------------


def render_f01(
    *,
    policy_results,
    output_dir,
    style=None,
    episode=None,
    annotations=None,
) -> FigureRecord:
    """F01: P0 / PI / P surfaces (3D) + contours, plus the canonical 3D set."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("F01", reason or REASON_NO_POLICY_SURFACES)
    axes_pair = grid_axes(result)
    if axes_pair is None:
        return _skip("F01", REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    canonical, reason = _canonical_surfaces(result)
    if canonical is None:
        return _skip("F01", reason or REASON_NO_POLICY_SURFACES)

    style = resolve_style(style, [result])
    panels = []
    order = ("P0", "PI", "P", "Q_claim", "Grid bp0*", "Grid bpI*")
    for name in order:
        panels.append(
            _Panel(
                kind="3d",
                title=f"3D {name}",
                family=_family_of(name),
                values=canonical[name],
                options={"b_values": b_values, "z_values": z_values},
            )
        )
    region = _default_region(result, canonical["P0"].shape)
    overlays = _boundary_overlay(result, style)
    for name in ("P0", "PI", "P"):
        panels.append(
            _Panel(
                kind="contour",
                title=f"等高线 {name} + 违约边界",
                family=_family_of(name),
                values=canonical[name],
                options={
                    "b_values": b_values,
                    "z_values": z_values,
                    "region": region,
                    "overlays": overlays,
                },
            )
        )

    n_samples = int(sum(_n_finite(canonical[name]) for name in order))
    meta = _base_meta(
        episode=episode_key,
        eta="η=1",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="否（曲面在固定 bank 网格上求值）",
        n_samples=n_samples,
        extra={
            "canonical_3d": list(CANONICAL_3D_NAMES),
            "grid_shape": list(canonical["P0"].shape),
        },
    )
    return _render_grid(
        "F01",
        panels,
        rows=3,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F01"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations,
            episode=episode_key,
            eta="η=1",
            i_slice=_i_slice_text(result),
            data_source=str(meta["data_source"]),
            capital_weighted="否",
            n_samples=n_samples,
        ),
    )


def render_f02(
    *,
    policy_results,
    output_dir,
    style=None,
    episode=None,
    annotations=None,
) -> FigureRecord:
    """F02: default boundary / default probability (Gomes-style 2D)."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("F02", reason or REASON_NO_POLICY_SURFACES)
    axes_pair = grid_axes(result)
    if axes_pair is None:
        return _skip("F02", REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    style = resolve_style(style, [result])

    bar_z = _surface(result, "eta1_bar_z")
    phat = _surface(result, "eta1_Phat")
    default_table = _pick_table(result, "default_boundary")
    hard_soft = _pick_table(result, "hard_soft_boundary")
    if bar_z is None and phat is None and default_table is None:
        return _skip("F02", REASON_NO_DEFAULT_BOUNDARY)

    shape = (b_values.size, z_values.size)
    region = _default_region(result, shape)
    overlays = _boundary_overlay(result, style)
    panels = []
    if bar_z is not None:
        panels.append(
            _Panel(
                kind="heatmap",
                title="bar_z = 1-sigmoid(Phat/τz)（软违约概率）",
                family="prob",
                values=bar_z,
                options={
                    "b_values": b_values,
                    "z_values": z_values,
                    "region": region,
                    "overlays": overlays,
                },
            )
        )
    if phat is not None:
        panels.append(
            _Panel(
                kind="heatmap",
                title="Phat(b,z)（股权值判别；≤0 为硬违约）",
                family="value",
                values=phat,
                options={
                    "b_values": b_values,
                    "z_values": z_values,
                    "region": region,
                    "overlays": overlays,
                },
            )
        )

    boundary_series = []
    if default_table is not None and {"b", "z_default"}.issubset(default_table.columns):
        boundary_series.append(
            {
                "x": _table_column(default_table, "b"),
                "y": _table_column(default_table, "z_default"),
                "label": "Phat=0（硬）",
                "color": "black",
            }
        )
    if hard_soft is not None and {"b", "z_soft_barz0p5"}.issubset(hard_soft.columns):
        boundary_series.append(
            {
                "x": _table_column(hard_soft, "b"),
                "y": _table_column(hard_soft, "z_soft_barz0p5"),
                "label": "bar_z=0.5（软）",
                "color": "tab:red",
                "linestyle": "--",
            }
        )
    panels.append(
        _Panel(
            kind="line",
            title="违约边界 z_default(b)：硬 vs 软",
            xlabel="b = 杠杆率 (债务/资本)",
            ylabel="z = 企业生产率冲击（违约门槛）",
            options={"series": boundary_series},
        )
    )

    rows = _pick_table(result, "policy_rows")
    if rows is not None and {"b", "conditional_default_prob"}.issubset(rows.columns):
        panels.append(
            _Panel(
                kind="scatter",
                title="条件违约概率 vs 杠杆（来自生产率转移，非 bar_z）",
                xlabel="b = 杠杆率",
                ylabel="Pr(Phat_child ≤ 0 | b_next)",
                options={
                    "series": [
                        {
                            "x": _table_column(rows, "b"),
                            "y": _table_column(rows, "conditional_default_prob"),
                            "label": "条件违约概率",
                        }
                    ],
                    "diag": False,
                },
            )
        )
    else:
        panels.append(
            _Panel(
                kind="text",
                title="条件违约概率",
                options={
                    "text": "policy_rows 无 conditional_default_prob 列\n（本面板省略，不伪造空白图）"
                },
            )
        )

    status_counts = ""
    if default_table is not None and "boundary_status" in default_table.columns:
        counts = default_table["boundary_status"].value_counts().to_dict()
        status_counts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    n_samples = int(shape[0] * shape[1])
    meta = _base_meta(
        episode=episode_key,
        eta="η=1",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
        extra={"boundary_status_counts": status_counts},
    )
    return _render_grid(
        "F02",
        panels,
        rows=2,
        cols=2,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F02"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations,
            episode=episode_key,
            eta="η=1",
            i_slice=_i_slice_text(result),
            data_source=str(meta["data_source"]),
            capital_weighted="否",
            n_samples=n_samples,
        ),
    )


def render_f03(
    *,
    policy_results,
    output_dir,
    style=None,
    episode=None,
    annotations=None,
) -> FigureRecord:
    """F03: Q Claim / Q Unit / Recovery / Q Effective (3D + default regions)."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("F03", reason or REASON_NO_POLICY_SURFACES)
    axes_pair = grid_axes(result)
    if axes_pair is None:
        return _skip("F03", REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    canonical, reason = _canonical_surfaces(result)
    if canonical is None:
        return _skip("F03", reason or REASON_NO_POLICY_SURFACES)
    q_claim = _surface(result, "eta1_Q_claim")
    q_unit = _surface(result, "eta1_q_unit")
    recovery = _surface(result, "eta1_recovery")
    if q_claim is None or q_unit is None or recovery is None:
        return _skip("F03", REASON_NO_Q_SURFACES)
    q_effective = _surface(result, "eta1_Q_effective")

    style = resolve_style(style, [result])
    shape = canonical["P0"].shape
    region = _default_region(result, shape)

    panels = [
        _Panel(kind="3d", title="3D Q_claim（Q Claim 发行定价口径）", family="q", values=q_claim,
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D q_unit（单位债券价）", family="q", values=q_unit,
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D Recovery（违约回收值）", family="q", values=recovery,
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D Q Effective（结算口径）", family="q",
               values=q_effective if q_effective is not None else np.full(shape, np.nan),
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D P0（canonical）", family="value", values=canonical["P0"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D PI（canonical）", family="value", values=canonical["PI"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D P（canonical）", family="value", values=canonical["P"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D Grid bp0*（canonical）", family="bp", values=canonical["Grid bp0*"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D Grid bpI*（canonical）", family="bp", values=canonical["Grid bpI*"],
               options={"b_values": b_values, "z_values": z_values}),
    ]
    for name, values in (("Q Claim", q_claim), ("q_unit", q_unit), ("Recovery", recovery)):
        panels.append(
            _Panel(
                kind="contour",
                title=f"{name} + Phat≤0 违约区",
                family="q",
                values=values,
                options={"b_values": b_values, "z_values": z_values, "region": region,
                         "overlays": _boundary_overlay(result, style)},
            )
        )

    n_samples = int(sum(_n_finite(item.values) for item in panels if item.values is not None))
    meta = _base_meta(
        episode=episode_key,
        eta="η=1",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
        extra={"q_calibers": ["Q_claim", "q_unit", "Q_effective", "recovery"],
               "canonical_3d": list(CANONICAL_3D_NAMES)},
    )
    return _render_grid(
        "F03",
        panels,
        rows=4,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F03"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=episode_key, eta="η=1", i_slice=_i_slice_text(result),
            data_source=str(meta["data_source"]), capital_weighted="否", n_samples=n_samples,
        ),
    )


def render_f04(
    *,
    policy_results,
    output_dir,
    style=None,
    episode=None,
    annotations=None,
) -> FigureRecord:
    """F04: Grid BP policy and investment boundary (3D + 2D)."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("F04", reason or REASON_NO_POLICY_SURFACES)
    axes_pair = grid_axes(result)
    if axes_pair is None:
        return _skip("F04", REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    canonical, reason = _canonical_surfaces(result)
    if canonical is None:
        return _skip("F04", reason or REASON_NO_POLICY_SURFACES)
    i_star = _surface(result, "eta1_investment_i_star")
    investment_table = _pick_table(result, "investment_boundary")
    if i_star is None and investment_table is None:
        return _skip("F04", REASON_NO_INVESTMENT_BOUNDARY)

    style = resolve_style(style, [result])
    shape = canonical["P0"].shape
    region = _default_region(result, shape)

    panels = [
        _Panel(kind="3d", title="3D Grid bp0*（不投资分支）", family="bp", values=canonical["Grid bp0*"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D Grid bpI*（投资分支）", family="bp", values=canonical["Grid bpI*"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D 投资边界 i*(b,z)", family="bp",
               values=i_star if i_star is not None else np.full(shape, np.nan),
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D P0（canonical）", family="value", values=canonical["P0"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D PI（canonical）", family="value", values=canonical["PI"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D P（canonical）", family="value", values=canonical["P"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D Q_claim（canonical）", family="q", values=canonical["Q_claim"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="contour", title="Grid bp0* + 违约边界", family="bp", values=canonical["Grid bp0*"],
               options={"b_values": b_values, "z_values": z_values, "region": region,
                        "overlays": _boundary_overlay(result, style)}),
        _Panel(kind="contour", title="Grid bpI* + 违约边界", family="bp", values=canonical["Grid bpI*"],
               options={"b_values": b_values, "z_values": z_values, "region": region,
                        "overlays": _boundary_overlay(result, style)}),
        _Panel(kind="contour", title="投资边界 i*(b,z)（PI-V0 零点）", family="bp",
               values=i_star if i_star is not None else np.full(shape, np.nan),
               options={"b_values": b_values, "z_values": z_values, "region": region,
                        "overlays": _boundary_overlay(result, style)}),
        _Panel(kind="heatmap", title="Bar_i_model（投资意愿）", family="prob",
               values=_or_nan(_surface(result, "eta1_bar_i_cond"), shape),
               options={"b_values": b_values, "z_values": z_values, "region": region}),
        _Panel(kind="heatmap", title="Bar_i_executed（执行投资）", family="prob",
               values=_or_nan(_surface(result, "eta1_bar_i_eff"), shape),
               options={"b_values": b_values, "z_values": z_values, "region": region}),
        _Panel(kind="contour", title="Grid bpI* - bp0*（投资分支杠杆溢价）", family="residual",
               values=_or_nan(canonical["Grid bpI*"] - canonical["Grid bp0*"], shape),
               options={"b_values": b_values, "z_values": z_values, "region": region,
                        "overlays": _boundary_overlay(result, style)}),
        _Panel(kind="heatmap", title="存活区指示（Phat>0）", family="prob",
               values=_or_nan((1.0 - region.astype(np.float64)) if region is not None else None, shape),
               options={"b_values": b_values, "z_values": z_values,
                        "overlays": _boundary_overlay(result, style)}),
    ]

    if investment_table is not None and {"b", "z", "investment_status_label"}.issubset(
        investment_table.columns
    ):
        labels = sorted(set(investment_table["investment_status_label"].astype(str)))
        encoded = (
            investment_table["investment_status_label"]
            .astype(str)
            .map({label: index for index, label in enumerate(labels)})
            .to_numpy(dtype=np.float64)
        )
        pivoted = np.full(shape, np.nan)
        b_column = _table_column(investment_table, "b")
        z_column = _table_column(investment_table, "z")
        if b_column is not None and z_column is not None:
            b_lookup = {float(value): index for index, value in enumerate(b_values)}
            z_lookup = {float(value): index for index, value in enumerate(z_values)}
            for row in range(len(encoded)):
                i_pos = b_lookup.get(float(b_column[row]))
                j_pos = z_lookup.get(float(z_column[row]))
                if i_pos is not None and j_pos is not None:
                    pivoted[i_pos, j_pos] = encoded[row]
        panels.append(
            _Panel(
                kind="heatmap",
                title="投资边界分类（0=" + ",1=".join(labels) + "）",
                family="bp",
                values=pivoted,
                options={"b_values": b_values, "z_values": z_values, "region": region},
            )
        )

    n_samples = int(sum(_n_finite(item.values) for item in panels if item.values is not None))
    status_counts = {}
    if investment_table is not None and "investment_status_label" in investment_table.columns:
        status_counts = (
            investment_table["investment_status_label"].astype(str).value_counts().to_dict()
        )
    meta = _base_meta(
        episode=episode_key,
        eta="η=1",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
        extra={"canonical_3d": list(CANONICAL_3D_NAMES),
               "investment_status_counts": status_counts},
    )
    return _render_grid(
        "F04",
        panels,
        rows=5,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F04"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=episode_key, eta="η=1", i_slice=_i_slice_text(result),
            data_source=str(meta["data_source"]), capital_weighted="否", n_samples=n_samples,
        ),
    )


def render_f05(
    *,
    bellman_results,
    output_dir,
    style=None,
    annotations=None,
) -> FigureRecord:
    """F05: Bellman residual vs episode (line + P90)."""
    episodes = _as_episode_map(bellman_results)
    if not episodes:
        return _skip("F05", REASON_NO_BELLMAN_METRICS)
    x = np.arange(len(episodes), dtype=np.float64)
    keys = list(episodes)
    styles = (
        ("fresh_p0_normalized", "fresh P0 残差（normalized）"),
        ("fresh_pi_normalized", "fresh PI 残差（normalized）"),
        ("frozen_p0_normalized", "frozen P0 残差（normalized）"),
        ("frozen_pi_normalized", "frozen PI 残差（normalized）"),
        ("fresh_p0_physical", "fresh P0 残差（physical）"),
        ("fresh_pi_physical", "fresh PI 残差（physical）"),
    )
    panels = []
    total = 0
    for prefix, label in styles:
        series = []
        for suffix, style_name, line in (
            ("mae", "MAE", "-"),
            ("p90", "P90", "--"),
            ("p99", "P99", ":"),
        ):
            values = []
            for key in keys:
                metrics = _result_metrics(episodes[key])
                values.append(float(metrics.get(f"{prefix}_{suffix}", float("nan"))))
            array = np.asarray(values, dtype=np.float64)
            if np.isfinite(array).any():
                total += int(np.isfinite(array).sum())
                series.append({"x": x, "y": array, "label": style_name, "linestyle": line})
        if series:
            panels.append(
                _Panel(
                    kind="line",
                    title=label,
                    xlabel="Episode（训练集数）",
                    ylabel="残差 |R|（normalized=R/(1+exp(clamp(x+z)))）",
                    options={"series": series, "logy": "symlog"},
                )
            )
    if not panels:
        return _skip("F05", REASON_NO_BELLMAN_METRICS)
    while len(panels) < 6:
        panels.append(_Panel(kind="text", title="（无数据）", options={"text": "该口径未报残差统计"}))

    counts = []
    for key in keys:
        metrics = _result_metrics(episodes[key])
        counts.append(int(metrics.get("fresh_p0_normalized_n_finite", 0.0)))
    n_samples = int(sum(counts)) if sum(counts) else total
    meta = _base_meta(
        episode=f"{keys[0]}..{keys[-1]}",
        eta="η∈{0,1} 混合（residual_stat_block 全体）",
        i_slice="i=bank 全体",
        data_source=_describe_bank_source(getattr(episodes[keys[-1]], "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
        extra={
            "episodes": [key for key in keys],
            "residual_definition": (
                "frozen=训练冻结策略 Bellman 恒等式残差; "
                "fresh=当前模型重新贪心求解的最优性残差"
            ),
        },
    )
    return _render_grid(
        "F05",
        panels,
        rows=2,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F05"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations,
            episode=meta["episode"],
            eta="η∈{0,1}",
            i_slice="bank 全体",
            data_source=str(meta["data_source"]),
            capital_weighted="否",
            n_samples=n_samples,
        ),
    )


def _aligned_policy_rows(results):
    frames = {}
    length = None
    index = None
    for key, result in results.items():
        frame = _pick_table(result, "policy_rows")
        if frame is None or frame.empty:
            return None
        frames[key] = frame
        if length is None:
            length = len(frame)
            index = frame["row_index"].to_numpy() if "row_index" in frame.columns else None
        else:
            if len(frame) != length:
                return None
            if index is not None and "row_index" in frame.columns:
                if not np.array_equal(index, frame["row_index"].to_numpy()):
                    return None
    return frames if frames else None


def render_f06(
    *,
    policy_results,
    output_dir,
    style=None,
    comparison=None,
    annotations=None,
) -> FigureRecord:
    """F06: P / Q / policy drift across episodes (multi-panel line)."""
    episodes = _as_episode_map(policy_results)
    if len(episodes) < 2:
        return _skip("F06", REASON_INSUFFICIENT_EPISODES)
    frames = _aligned_policy_rows(episodes)
    if frames is None:
        return _skip("F06", REASON_POLICY_ROWS_NOT_ALIGNED)
    keys = list(frames)
    x = np.arange(1, len(keys), dtype=np.float64)

    def _drift(column):
        values = []
        for previous, current in zip(keys[:-1], keys[1:]):
            left = _table_column(frames[previous], column)
            right = _table_column(frames[current], column)
            if left is None or right is None:
                values.append(np.nan)
            else:
                values.append(float(np.nanmean(np.abs(right - left))))
        array = np.asarray(values, dtype=np.float64)
        return array if np.isfinite(array).any() else None

    panels = []
    drift_series = {}
    for column in ("P", "P0", "PI", "q_issue_at_star_p0", "bp_star_mix", "b_next_mix"):
        array = _drift(column)
        if array is not None:
            drift_series[column] = array
    if not drift_series:
        return _skip("F06", REASON_NO_POLICY_ROWS)
    for column, array in drift_series.items():
        panels.append(
            _Panel(
                kind="line",
                title=f"{column} 跨集漂移",
                xlabel="Episode e（相对 e-1 的差分）",
                ylabel="mean |f_e - f_{e-1}|（固定 bank 状态逐行）",
                family=_family_of(column),
                options={"series": [{"x": x, "y": array, "label": column}]},
            )
        )

    flip_series = []
    for name, label in (
        ("policy_flip_rate", "policy_flip_rate"),
        ("default_flip_rate", "default_flip_rate"),
        ("investment_flip_rate", "investment_flip_rate"),
    ):
        values = []
        for previous, current in zip(keys[:-1], keys[1:]):
            left = frames[previous]
            right = frames[current]
            if name == "policy_flip_rate":
                if not {"bp_star_mix", "eta"}.issubset(left.columns) or "bp_star_mix" not in right.columns:
                    values.append(np.nan)
                    continue
                eta = _table_column(left, "eta")
                diagnostics = getattr(episodes[current], "diagnostics", None) or {}
                tol = float(diagnostics.get("policy_flip_tol", 1e-2))
                mask = (
                    np.asarray(eta, dtype=np.float64) > 0.5
                    if eta is not None
                    else np.ones(len(left), dtype=bool)
                )
                delta = np.abs(
                    _table_column(right, "bp_star_mix") - _table_column(left, "bp_star_mix")
                )
                values.append(float(np.mean(delta[mask] > tol)) if mask.any() else np.nan)
            elif name == "default_flip_rate":
                if "Phat" not in left.columns or "Phat" not in right.columns:
                    values.append(np.nan)
                    continue
                values.append(
                    float(
                        np.mean(
                            (_table_column(left, "Phat") <= 0.0)
                            != (_table_column(right, "Phat") <= 0.0)
                        )
                    )
                )
            else:
                if not {"PI", "P0"}.issubset(left.columns) or not {"PI", "P0"}.issubset(right.columns):
                    values.append(np.nan)
                    continue
                left_dec = _table_column(left, "PI") > _table_column(left, "P0")
                right_dec = _table_column(right, "PI") > _table_column(right, "P0")
                values.append(float(np.mean(left_dec != right_dec)))
        array = np.asarray(values, dtype=np.float64)
        if np.isfinite(array).any():
            flip_series.append({"x": x, "y": array, "label": label})
    if flip_series:
        panels.append(
            _Panel(
                kind="line",
                title="决策翻转率（flip rate）",
                xlabel="Episode e（相对 e-1 的差分）",
                ylabel="翻转比例（阈值 policy_flip_tol）",
                options={"series": flip_series},
            )
        )

    comparison_series = []
    if isinstance(comparison, pd.DataFrame) and {"episode", "mean_abs_diff"}.issubset(
        comparison.columns
    ):
        group_column = "surface" if "surface" in comparison.columns else comparison.columns[0]
        for surface_name, group in comparison.groupby(group_column):
            comparison_series.append(
                {
                    "x": pd.to_numeric(group["episode"], errors="coerce").to_numpy(dtype=np.float64),
                    "y": pd.to_numeric(group["mean_abs_diff"], errors="coerce").to_numpy(dtype=np.float64),
                    "label": str(surface_name),
                }
            )
    if comparison_series:
        panels.append(
            _Panel(
                kind="line",
                title="comparator 工件漂移（comparison）",
                xlabel="Episode",
                ylabel="mean_abs_diff",
                options={"series": comparison_series},
            )
        )
    else:
        panels.append(
            _Panel(
                kind="text",
                title="comparator 输入",
                options={
                    "text": "comparison=None 或格式不适用\n（本面板省略，漂移由 policy_rows 逐行重算）"
                },
            )
        )

    while len(panels) < 9:
        panels.append(_Panel(kind="text", title="（无数据）", options={"text": "该漂移口径不可用"}))

    n_samples = int(len(next(iter(frames.values())))) * max(len(keys) - 1, 1)
    meta = _base_meta(
        episode=f"{keys[0]}..{keys[-1]}",
        eta="η∈{0,1}（flip 统计在 η=1 行）",
        i_slice="bank 全体",
        data_source=_describe_bank_source(getattr(episodes[keys[-1]], "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
        extra={
            "episodes": [key for key in keys],
            "drift_definition": "同一固定 bank 状态逐行 |f_e - f_{e-1}| 的均值",
        },
    )
    return _render_grid(
        "F06",
        panels,
        rows=3,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F06"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=meta["episode"], eta="η∈{0,1}", i_slice="bank 全体",
            data_source=str(meta["data_source"]), capital_weighted="否", n_samples=n_samples,
        ),
    )


def _macro_pairs(result):
    frame = _pick_table(result, "fc1_one_step_pairs")
    if frame is None or frame.empty:
        return None
    if not {"hatcf_t", "hatc_t", "lnkf_t", "lnk_t"}.issubset(frame.columns):
        return None
    return frame


def render_f07(
    *,
    macro_results,
    output_dir,
    style=None,
    annotations=None,
) -> FigureRecord:
    """F07: FC1 forecast vs realized (scatter + path)."""
    episodes = _as_episode_map(macro_results)
    if not episodes:
        return _skip("F07", REASON_NO_MACRO_PAIRS)
    key = list(episodes)[-1]
    result = episodes[key]
    frame = _macro_pairs(result)
    if frame is None:
        return _skip("F07", REASON_NO_MACRO_PAIRS)

    style = resolve_style(style, [result])
    hatc_x = _table_column(frame, "hatc_t")
    hatc_y = _table_column(frame, "hatcf_t")
    lnk_x = _table_column(frame, "lnk_t")
    lnk_y = _table_column(frame, "lnkf_t")

    panels = [
        _Panel(
            kind="scatter",
            title="Hatcf_t vs Hatc_t（一步预测 vs 实现，shift=0）",
            xlabel="Hatc_t = log(C/K) 实现聚合",
            ylabel="Hatcf_t = 对 t 的信念预测",
            options={"series": [{"x": hatc_x, "y": hatc_y, "label": "Bank D 宏观持有期"}]},
        ),
        _Panel(
            kind="scatter",
            title="LnKF_t vs LnK_t（资本信念 vs 实现）",
            xlabel="LnK_t = log(ΣK) 实现聚合",
            ylabel="LnKF_t = 对 t 的信念预测",
            options={"series": [{"x": lnk_x, "y": lnk_y, "label": "Bank D 宏观持有期"}]},
        ),
    ]

    path_series_hatc = []
    path_series_lnk = []
    if "path" in frame.columns and "t" in frame.columns:
        path_column = pd.to_numeric(frame["path"], errors="coerce")
        paths = sorted(path_column.dropna().unique())[:4]
        for path_id in paths:
            subset = frame.loc[path_column == path_id].sort_values("t", kind="mergesort")
            t_values = _table_column(subset, "t")
            path_series_hatc.append(
                {"x": t_values, "y": _table_column(subset, "hatc_t"), "label": f"path {int(path_id)} 实现"}
            )
            path_series_hatc.append(
                {"x": t_values, "y": _table_column(subset, "hatcf_t"),
                 "label": f"path {int(path_id)} 预测", "linestyle": "--"}
            )
            path_series_lnk.append(
                {"x": t_values, "y": _table_column(subset, "lnk_t"), "label": f"path {int(path_id)} 实现"}
            )
            path_series_lnk.append(
                {"x": t_values, "y": _table_column(subset, "lnkf_t"),
                 "label": f"path {int(path_id)} 预测", "linestyle": "--"}
            )
    panels.append(
        _Panel(
            kind="line",
            title="Hatcf vs Hatc 路径（前 4 条 path）",
            xlabel="t（宏观节点）",
            ylabel="Hatc / Hatcf",
            options={"series": path_series_hatc},
        )
    )
    panels.append(
        _Panel(
            kind="line",
            title="LnKF vs LnK 路径（前 4 条 path）",
            xlabel="t（宏观节点）",
            ylabel="LnK / LnKF",
            options={"series": path_series_lnk},
        )
    )

    stats_text = []
    for label, calculated, forecast in (("hatc", hatc_x, hatc_y), ("lnk", lnk_x, lnk_y)):
        mask = np.isfinite(calculated) & np.isfinite(forecast)
        if mask.any():
            err = forecast[mask] - calculated[mask]
            stats_text.append(
                f"{label}: n={int(mask.sum())}  MAE={np.mean(np.abs(err)):.4g}  "
                f"Bias={np.mean(err):.4g}  RMSE={np.sqrt(np.mean(err ** 2)):.4g}"
            )
    panels.append(
        _Panel(kind="text", title="一步预测误差汇总", options={"text": "\n".join(stats_text) or "无数据"})
    )
    panels.append(
        _Panel(
            kind="text",
            title="口径说明",
            options={
                "text": (
                    "hatcf/lnkf 是信念（firm 状态第 5/6 列）\n"
                    "Hatc/LnK 是实现聚合\n"
                    "同行 hatcf_t vs Hatc_t 即 shift=0 一步预测\n"
                    "配对只走 (path,t,branch,economic_node_id) 键"
                )
            },
        )
    )

    n_samples = int(len(frame))
    meta = _base_meta(
        episode=key,
        eta="不适用（宏观行）",
        i_slice="不适用",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
        extra={"pair_keys": ["path", "t", "branch", "economic_node_id"], "shift": 0},
    )
    return _render_grid(
        "F07",
        panels,
        rows=2,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F07"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=key, data_source=str(meta["data_source"]),
            capital_weighted="否", n_samples=n_samples,
        ),
    )


def _anchor_path_tables(result):
    tables = _result_tables(result)
    return {
        str(key): frame
        for key, frame in tables.items()
        if str(key).startswith("fc1_anchor_recursion_")
        and isinstance(frame, pd.DataFrame)
        and not frame.empty
    }


def render_f08(
    *,
    macro_results,
    output_dir,
    style=None,
    annotations=None,
) -> FigureRecord:
    """F08: macro dynamics / SDF time series."""
    episodes = _as_episode_map(macro_results)
    if not episodes:
        return _skip("F08", REASON_NO_MACRO_ANCHOR_PATHS)
    key = list(episodes)[-1]
    result = episodes[key]
    style = resolve_style(style, [result])

    anchors = _anchor_path_tables(result)
    sdf_table = _pick_table(result, "sdf_macro_state_summary")
    if not anchors and sdf_table is None:
        return _skip("F08", REASON_NO_MACRO_ANCHOR_PATHS)

    panels = []
    n_samples = 0
    for name, frame in sorted(anchors.items()):
        panels.append(
            _Panel(
                kind="line",
                title=f"宏观锚点递归路径 {name.replace('fc1_anchor_recursion_', '')}",
                xlabel="递归步数 h",
                ylabel="预测值（hatc_pred / lnk_pred）",
                options={
                    "series": [
                        {"x": _table_column(frame, "step"), "y": _table_column(frame, "hatc_pred"),
                         "label": "hatc_pred"},
                        {"x": _table_column(frame, "step"), "y": _table_column(frame, "lnk_pred"),
                         "label": "lnk_pred", "linestyle": "--"},
                    ]
                },
            )
        )
        n_samples += int(len(frame))

    if sdf_table is not None and {"macro_state", "M_mean"}.issubset(sdf_table.columns):
        labels = sdf_table["macro_state"].astype(str).tolist()
        panels.append(
            _Panel(
                kind="bar",
                title="SDF M 条件均值（按宏观态）",
                xlabel="宏观态（x 分位：Recession/Normal/Expansion）",
                ylabel="E[M | 宏观态]（± 标准误）",
                options={"labels": labels, "values": _table_column(sdf_table, "M_mean"),
                         "yerr": _table_column(sdf_table, "M_mean_se"), "rotation": 15},
            )
        )
        panels.append(
            _Panel(
                kind="bar",
                title="归一化财富 Euler 残差（按宏观态）",
                xlabel="宏观态",
                ylabel="mean residual（± 标准误）",
                options={"labels": labels, "values": _table_column(sdf_table, "mean_residual"),
                         "yerr": _table_column(sdf_table, "se_residual"), "rotation": 15},
            )
        )
        n_samples += int(len(sdf_table))

    metrics = _result_metrics(result)
    lines = []
    for name in (
        "sdf_M_mean", "sdf_M_mean_se", "sdf_M_var", "sdf_signed_aio_estimate",
        "sdf_signed_aio_se", "sdf_population_target_estimate", "sdf_population_target_se",
        "sdf_g_mean_low", "sdf_g_mean_high", "sdf_g_var_high", "sdf_g_max",
        "sdf_nonfinite_ratio", "sdf_log_R_clip_share", "sdf_valid_parent_ratio",
    ):
        if name in metrics:
            lines.append(f"{name} = {float(metrics[name]):.6g}")
    panels.append(
        _Panel(kind="text", title="SDF 数值安全与矩约束", options={"text": "\n".join(lines) or "无 SDF 指标"})
    )
    panels.append(
        _Panel(
            kind="text",
            title="口径护栏",
            options={
                "text": (
                    "Signed AiO 样本均值 ≠ 总体条件矩\n"
                    "总体目标另报 sdf_population_target_*\n"
                    "矩约束 g_* 与 AiO 分开报告，不混池"
                )
            },
        )
    )
    while len(panels) < 9:
        panels.append(_Panel(kind="text", title="（无数据）", options={"text": "该面板数据缺失"}))

    meta = _base_meta(
        episode=key,
        eta="不适用（宏观行）",
        i_slice="不适用",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
        extra={"macro_anchors": sorted(anchors), "sdf_table_present": sdf_table is not None},
    )
    return _render_grid(
        "F08",
        panels,
        rows=3,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F08"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=key, data_source=str(meta["data_source"]),
            capital_weighted="否", n_samples=n_samples,
        ),
    )


def _macro_state_labels(x_values, quantiles=(1.0 / 3.0, 2.0 / 3.0)):
    array = np.asarray(x_values, dtype=np.float64)
    out = np.empty(array.shape[0], dtype=object)
    out[:] = ""
    finite = np.isfinite(array)
    if not finite.any():
        return out.astype(str)
    edges = [float(np.quantile(array[finite], float(q))) for q in quantiles]
    bins = np.digitize(array, edges, right=True)
    labels = ("recession", "normal", "expansion")
    for index, label in enumerate(labels):
        out[bins == index] = label
    out[~finite] = ""
    return out.astype(str)


def _sim_firm_panel(result):
    frame = _pick_table(result, "firm_panel")
    if frame is None or frame.empty:
        return None
    if not {"b", "x"}.issubset(frame.columns):
        return None
    return frame


def render_f09(
    *,
    sim_results,
    output_dir,
    style=None,
    annotations=None,
) -> FigureRecord:
    """F09: leverage distribution by macro state (Gomes-style histogram)."""
    episodes = _as_episode_map(sim_results)
    if not episodes:
        return _skip("F09", REASON_NO_SIM_PANEL)
    key = list(episodes)[-1]
    result = episodes[key]
    panel_frame = _sim_firm_panel(result)
    if panel_frame is None:
        return _skip("F09", REASON_NO_SIM_PANEL)

    style = resolve_style(style, [result])
    b_values = _table_column(panel_frame, "b")
    x_values = _table_column(panel_frame, "x")
    k_values = _table_column(panel_frame, "K")
    if b_values is None or x_values is None:
        return _skip("F09", REASON_NO_SIM_PANEL)
    if k_values is None:
        k_values = np.ones_like(b_values)

    t_column = _table_column(panel_frame, "t")
    states = _macro_state_labels(x_values)
    slices = []
    if t_column is not None and np.isfinite(t_column).any():
        unique = sorted(set(_finite(t_column).tolist()))
        if unique:
            picks = {unique[0], unique[-1]}
            if len(unique) > 2:
                picks.add(unique[len(unique) // 2])
            slices = sorted(picks)
    if not slices:
        slices = [float("nan")]

    b_low = float(style.b_range[0]) if style.b_range is not None else float(np.nanmin(b_values))
    b_high = float(style.b_range[1]) if style.b_range is not None else float(np.nanmax(b_values))
    if not math.isfinite(b_low) or not math.isfinite(b_high) or b_high <= b_low:
        good = _finite(b_values)
        if good.size == 0:
            return _skip("F09", REASON_NO_FINITE_SAMPLES)
        b_low, b_high = float(np.min(good)), float(np.max(good))
        if b_high <= b_low:
            b_high = b_low + 1e-9
    edges = np.linspace(b_low, b_high, 21)

    panels = []
    n_samples = 0
    for caliber, weights, label in (
        ("count", np.ones_like(b_values), "计数加权"),
        ("capital", k_values, "资本加权"),
    ):
        for state in ("recession", "normal", "expansion"):
            series = []
            for slice_value in slices:
                mask = states == state
                if t_column is not None and math.isfinite(slice_value):
                    mask = mask & np.isclose(t_column, slice_value)
                subset = b_values[mask]
                w = weights[mask]
                if subset.size == 0:
                    continue
                hist, bin_edges = np.histogram(subset, bins=edges, weights=w)
                total = float(np.nansum(hist))
                if total <= 0:
                    continue
                hist = hist / total
                centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                series.append(
                    {
                        "x": centers,
                        "y": hist,
                        "label": f"t={slice_value:g}" if math.isfinite(slice_value) else "全部 t",
                    }
                )
                n_samples += int(np.isfinite(subset).sum())
            panels.append(
                _Panel(
                    kind="hist",
                    title=f"{state}｜{label}",
                    xlabel="b = 杠杆率 (债务/资本)",
                    ylabel="分布质量（归一化）",
                    options={"series": series, "fill": caliber == "count"},
                )
            )
    if not any(item.options.get("series") for item in panels):
        return _skip("F09", REASON_NO_FINITE_SAMPLES)

    meta = _base_meta(
        episode=key,
        eta="不适用（模拟截面）",
        i_slice="不适用",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="计数加权与资本加权双口径（上行计数、下行资本）",
        extinct_paths=_describe_extinct(result),
        n_samples=n_samples,
        extra={"macro_state_rule": "x 分位 (1/3, 2/3) → recession/normal/expansion",
               "time_slices": [float(v) for v in slices]},
    )
    return _render_grid(
        "F09",
        panels,
        rows=2,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F09"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=key, data_source=str(meta["data_source"]),
            capital_weighted="双口径", extinct_paths=str(meta["extinct_paths"]),
            n_samples=n_samples,
        ),
    )


def render_f10(
    *,
    sim_results,
    output_dir,
    style=None,
    annotations=None,
) -> FigureRecord:
    """F10: b-z joint distribution over time (heatmap + marginals)."""
    episodes = _as_episode_map(sim_results)
    if not episodes:
        return _skip("F10", REASON_NO_JOINT_HIST)
    key = list(episodes)[-1]
    result = episodes[key]
    hists = _joint_hists(result)
    if not hists:
        return _skip("F10", REASON_NO_JOINT_HIST)
    marginals = _pick_table(result, "distribution_marginals")
    style = resolve_style(style, [result])

    slots = sorted(hists)
    selected = slots[:6]
    rows = 2 if len(selected) > 3 else 1
    cols = 3 if len(selected) > 3 else max(len(selected), 1)

    _apply_style(style)
    fig = plt.figure(figsize=(5.0 * cols, 4.6 * rows))
    outer = fig.add_gridspec(rows, cols)
    drawn = 0
    used = []
    total_cells = 0
    for index, (t_slice, caliber) in enumerate(selected):
        row, col = divmod(index, cols)
        hist = np.asarray(hists[(t_slice, caliber)], dtype=np.float64)
        if hist.ndim != 2 or hist.size == 0:
            used.append({"title": f"t={t_slice}/{caliber}", "drawn": False})
            continue
        b_range = style.b_range or (0.0, 1.0)
        z_range = style.z_range or (-1.0, 1.0)
        nb, nz = hist.shape
        b_edges = np.linspace(float(b_range[0]), float(b_range[1]), nb + 1)
        z_edges = np.linspace(float(z_range[0]), float(z_range[1]), nz + 1)
        b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

        sub = outer[row, col].subgridspec(
            2, 2, width_ratios=[1.0, 0.24], height_ratios=[0.24, 1.0],
            wspace=0.08, hspace=0.08,
        )
        ax_top = fig.add_subplot(sub[0, 0])
        ax_main = fig.add_subplot(sub[1, 0], sharex=ax_top)
        ax_side = fig.add_subplot(sub[1, 1], sharey=ax_main)

        total = float(np.nansum(hist))
        normalised = hist / total if total > 0 else hist
        total_cells += int(round(total))
        mesh = ax_main.pcolormesh(
            b_edges, z_edges, normalised.T, shading="flat", cmap=_cmap(style.cmap_density)
        )
        fig.colorbar(mesh, ax=ax_main, fraction=0.046, pad=0.02)
        ax_main.set_xlabel("b = 杠杆率")
        ax_main.set_ylabel("z = 企业生产率冲击")
        ax_main.set_xlim(float(b_range[0]), float(b_range[1]))
        ax_main.set_ylim(float(z_range[0]), float(z_range[1]))

        ax_top.plot(b_centers, normalised.sum(axis=1), color="tab:blue")
        ax_top.fill_between(b_centers, normalised.sum(axis=1), alpha=0.25, color="tab:blue")
        ax_top.set_ylabel("b 边缘")
        ax_top.tick_params(labelbottom=False)
        ax_side.plot(normalised.sum(axis=0), z_centers, color="tab:orange")
        ax_side.fill_betweenx(z_centers, normalised.sum(axis=0), alpha=0.25, color="tab:orange")
        ax_side.set_xlabel("z 边缘")
        ax_side.tick_params(labelleft=False)

        if marginals is not None and {"t", "caliber", "axis", "bin_center", "weight"}.issubset(
            marginals.columns
        ):
            t_numeric = pd.to_numeric(marginals["t"], errors="coerce").to_numpy(dtype=np.float64)
            for axis_name, axis in (("b", ax_top), ("z", ax_side)):
                subset = marginals.loc[
                    (marginals["axis"] == axis_name)
                    & np.isclose(t_numeric, float(t_slice))
                    & (marginals["caliber"].astype(str) == str(caliber))
                ]
                if subset.empty:
                    continue
                centers = pd.to_numeric(subset["bin_center"], errors="coerce").to_numpy(dtype=np.float64)
                weights = pd.to_numeric(subset["weight"], errors="coerce").to_numpy(dtype=np.float64)
                if axis_name == "b":
                    axis.plot(centers, weights, color="black", linestyle="--", linewidth=1.0)
                else:
                    axis.plot(weights, centers, color="black", linestyle="--", linewidth=1.0)

        ax_top.set_title(
            f"t={t_slice}｜caliber={caliber}（{'资本加权' if caliber == 'capital' else '计数加权'}）"
        )
        drawn += 1
        used.append({"title": f"t={t_slice}/{caliber}", "drawn": True})

    if drawn == 0:
        plt.close(fig)
        return _skip("F10", REASON_NO_JOINT_HIST)

    fig.suptitle(f"F10: {CAPTIONS['F10'].split('。')[0]}", fontsize=style.title_size + 1.5)
    fig.subplots_adjust(left=0.06, right=0.955, top=0.90, bottom=0.115)
    footer = " | ".join(
        _annotation_lines(
            _annotation_dict(
                annotations,
                episode=key,
                data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
                capital_weighted="count/capital 双口径",
                extinct_paths=_describe_extinct(result),
                n_samples=total_cells,
            )
        )
    )
    fig.text(0.005, 0.005, footer, ha="left", va="bottom", fontsize=style.footer_size)
    path = Path(output_dir) / FIGURE_FILENAMES["F10"]
    _save(fig, path, style)

    meta = _base_meta(
        episode=key,
        eta="不适用（模拟截面）",
        i_slice="不适用",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="count/capital 双口径",
        extinct_paths=_describe_extinct(result),
        n_samples=total_cells,
        extra={"time_slices": [{"t": t, "caliber": c} for t, c in selected],
               "marginals_present": marginals is not None,
               "hist_shape": list(np.asarray(hists[selected[0]]).shape)},
    )
    meta.update(_style_meta(style))
    meta["panel_status"] = used
    return FigureRecord(
        figure_id="F10",
        path=str(path),
        caption=CAPTIONS["F10"],
        status="ok",
        missing_reason=None,
        n_samples=total_cells,
        meta=meta,
    )


def render_f11_waterfall(
    *,
    sim_results,
    output_dir,
    style=None,
    annotations=None,
) -> FigureRecord:
    """F11: t0->t1 capital waterfall + survival / default / active paths."""
    episodes = _as_episode_map(sim_results)
    if not episodes:
        return _skip("F11", REASON_NO_CAPITAL_WATERFALL)
    key = list(episodes)[-1]
    result = episodes[key]
    waterfall = _pick_table(result, "capital_waterfall")
    if waterfall is None or waterfall.empty:
        return _skip("F11", REASON_NO_CAPITAL_WATERFALL)
    period_counts = _pick_table(result, "period_counts")
    path_status = _pick_table(result, "path_status")
    if period_counts is None and path_status is None:
        return _skip("F11", REASON_NO_SURVIVAL_ACCOUNT)

    required = (
        "initial_capital", "ending_capital", "incumbent_dk",
        "entrant_capital", "exiting_incumbent_capital",
    )
    if not set(required).issubset(waterfall.columns):
        return _skip("F11", REASON_NO_CAPITAL_WATERFALL)

    style = resolve_style(style, [result])
    _apply_style(style)
    fig = plt.figure(figsize=(17.0, 9.0))
    grid = fig.add_gridspec(2, 3)

    totals = _pick_table(result, "capital_waterfall_totals")
    source = totals if totals is not None and not totals.empty else waterfall
    steps = []
    if {"t_from", "t_to"}.issubset(source.columns):
        t_from = pd.to_numeric(source["t_from"], errors="coerce").to_numpy(dtype=np.float64)
        t_to = pd.to_numeric(source["t_to"], errors="coerce").to_numpy(dtype=np.float64)
        steps = list(zip(t_from, t_to))
    if not steps:
        plt.close(fig)
        return _skip("F11", REASON_NO_CAPITAL_WATERFALL)

    n_samples = int(len(waterfall))
    for index, (t_from, t_to) in enumerate(steps[:3]):
        ax = fig.add_subplot(grid[0, index])
        if totals is not None and not totals.empty and index < len(source):
            row = source.iloc[index]
        else:
            t_from_col = pd.to_numeric(waterfall["t_from"], errors="coerce")
            t_to_col = pd.to_numeric(waterfall["t_to"], errors="coerce")
            subset = waterfall.loc[np.isclose(t_from_col, t_from) & np.isclose(t_to_col, t_to)]
            row = subset.iloc[0] if not subset.empty else None
        if row is None:
            ax.axis("off")
            continue
        labels = ["期初资本", "存续 ΔK", "进入资本", "退出资本", "记账残差", "期末资本"]
        values = [
            float(row["initial_capital"]),
            float(row["incumbent_dk"]),
            float(row["entrant_capital"]),
            -float(row["exiting_incumbent_capital"]),
            float(row.get("accounting_residual", 0.0)),
            float(row["ending_capital"]),
        ]
        running = 0.0
        bottoms = []
        heights = []
        colors = []
        for position, value in enumerate(values):
            if position in (0, len(values) - 1):
                bottoms.append(0.0)
                heights.append(value)
                colors.append("tab:blue")
                running = value
            else:
                if value >= 0:
                    bottoms.append(running)
                    heights.append(value)
                    colors.append("tab:green")
                else:
                    bottoms.append(running + value)
                    heights.append(-value)
                    colors.append("tab:red")
                running += value
        ax.bar(np.arange(len(labels)), heights, bottom=bottoms, color=colors, alpha=0.85)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("资本（K）")
        ax.set_title(f"资本 Waterfall t={int(t_from)}→{int(t_to)}")
        ax.grid(alpha=0.2, axis="y")
        residual = float(row.get("accounting_residual", float("nan")))
        ax.text(
            0.02, 0.96, f"恒等式残差 = {residual:.3e}",
            transform=ax.transAxes, va="top", fontsize=style.legend_size,
        )

    ax = fig.add_subplot(grid[1, 0])
    if period_counts is not None and {"t", "active_paths"}.issubset(period_counts.columns):
        t_values = _table_column(period_counts, "t")
        for column, label in (
            ("active_paths", "活跃 path"),
            ("extinct_paths", "灭绝 path"),
            ("n_alive_firms", "存活企业"),
            ("n_defaults", "违约/退出数"),
            ("n_entrants", "进入数"),
        ):
            values = _table_column(period_counts, column)
            if values is not None:
                ax.plot(t_values, values, marker="o", markersize=3, label=label)
        ax.set_xlabel("t（模拟期）")
        ax.set_ylabel("计数")
        ax.set_title("存活 / 违约 / 活跃 path 逐期账")
        ax.legend()
        ax.grid(alpha=0.25)
    elif path_status is not None and {"path", "extinct"}.issubset(path_status.columns):
        flags = path_status["extinct"].astype(bool).to_numpy()
        ax.bar(np.arange(2), [int((~flags).sum()), int(flags.sum())],
               color=["tab:green", "tab:red"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["存活 path", "灭绝 path"])
        ax.set_title("path 状态（灭绝 path 保留在统计中）")
        ax.grid(alpha=0.2, axis="y")

    ax = fig.add_subplot(grid[1, 1])
    if period_counts is not None:
        t_values = _table_column(period_counts, "t")
        for column, label in (
            ("n_defaults", "n_defaults"),
            ("n_entrants", "n_entrants"),
            ("n_investors", "n_investors"),
        ):
            values = _table_column(period_counts, column)
            if values is not None:
                ax.plot(t_values, values, marker="o", markersize=3, label=label)
        ax.set_xlabel("t")
        ax.set_ylabel("计数")
        ax.set_title("逐期事件计数")
        ax.legend()
        ax.grid(alpha=0.25)
    elif {"default_rate", "survival_rate"}.issubset(waterfall.columns):
        t_to = _table_column(waterfall, "t_to")
        ax.plot(t_to, _table_column(waterfall, "survival_rate"), marker="o", label="survival_rate")
        ax.plot(t_to, _table_column(waterfall, "default_rate"), marker="o", label="default_rate")
        ax.set_xlabel("t_to")
        ax.set_ylabel("比率")
        ax.set_title("存活率 / 违约率")
        ax.legend()
        ax.grid(alpha=0.25)

    ax = fig.add_subplot(grid[1, 2])
    if "accounting_residual" in waterfall.columns:
        t_to = _table_column(waterfall, "t_to")
        ax.plot(t_to, _table_column(waterfall, "accounting_residual"),
                marker="o", linestyle="none", markersize=3)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xlabel("t_to")
        ax.set_ylabel("accounting_residual")
        ax.set_title("资本恒等式残差（应贴近 0）")
        ax.grid(alpha=0.25)

    fig.suptitle(f"F11: {CAPTIONS['F11'].split('。')[0]}", fontsize=style.title_size + 1.5)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.135, hspace=0.55, wspace=0.30)
    footer = " | ".join(
        _annotation_lines(
            _annotation_dict(
                annotations,
                episode=key,
                data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
                capital_weighted="资本账本（总量口径）",
                extinct_paths=_describe_extinct(result),
                n_samples=n_samples,
            )
        )
    )
    fig.text(0.005, 0.005, footer, ha="left", va="bottom", fontsize=style.footer_size)
    path = Path(output_dir) / FIGURE_FILENAMES["F11"]
    _save(fig, path, style)

    meta = _base_meta(
        episode=key,
        eta="不适用（模拟截面）",
        i_slice="不适用",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="资本账本（总量口径）",
        extinct_paths=_describe_extinct(result),
        n_samples=n_samples,
        extra={
            "transitions": [[float(a), float(b)] for a, b in steps[:3]],
            "identity": "K_end-K_start = deltaK_incumbent + K_entry_endpoint - K_exit_old + residual",
            "survival_panel_source": "period_counts" if period_counts is not None else "path_status",
        },
    )
    meta.update(_style_meta(style))
    return FigureRecord(
        figure_id="F11",
        path=str(path),
        caption=CAPTIONS["F11"],
        status="ok",
        missing_reason=None,
        n_samples=n_samples,
        meta=meta,
    )


def render_f12(
    *,
    output_dir,
    bellman_results=None,
    policy_results=None,
    macro_results=None,
    sim_results=None,
    comparison=None,
    style=None,
    annotations=None,
) -> FigureRecord:
    """F12: general-equilibrium convergence dashboard."""
    blocks = {
        "bellman": _as_episode_map(bellman_results),
        "policy": _as_episode_map(policy_results),
        "macro": _as_episode_map(macro_results),
        "sim": _as_episode_map(sim_results),
    }
    present = [name for name, values in blocks.items() if values]
    if len(present) < 2:
        return _skip("F12", REASON_INSUFFICIENT_BLOCKS)
    style = resolve_style(style, [item for values in blocks.values() for item in values.values()])

    panels = []
    n_samples = 0

    bellman = blocks["bellman"]
    if bellman:
        keys = list(bellman)
        x = np.arange(len(keys), dtype=np.float64)
        series = []
        for metric, label in (
            ("fresh_p0_normalized_mae", "fresh P0 MAE"),
            ("fresh_pi_normalized_mae", "fresh PI MAE"),
            ("frozen_p0_normalized_mae", "frozen P0 MAE"),
            ("frozen_pi_normalized_mae", "frozen PI MAE"),
        ):
            array = np.asarray(
                [float(_result_metrics(bellman[key]).get(metric, float("nan"))) for key in keys],
                dtype=np.float64,
            )
            if np.isfinite(array).any():
                series.append({"x": x, "y": array, "label": label})
                n_samples += int(np.isfinite(array).sum())
        panels.append(
            _Panel(
                kind="line",
                title="Bellman 残差收敛",
                xlabel="Episode",
                ylabel="MAE（normalized）",
                options={"series": series, "logy": "symlog"},
            )
        )
    else:
        panels.append(_Panel(kind="text", title="Bellman", options={"text": "无 bellman 结果"}))

    policy = blocks["policy"]
    if len(policy) >= 2:
        keys = list(policy)
        drift_x = np.arange(1, len(keys), dtype=np.float64)
        series = []
        for column, label in (("P", "mean|ΔP|"), ("bp_star_mix", "mean|Δbp*|")):
            values = []
            for previous, current in zip(keys[:-1], keys[1:]):
                left = _table_column(_pick_table(policy[previous], "policy_rows"), column)
                right = _table_column(_pick_table(policy[current], "policy_rows"), column)
                values.append(
                    float(np.nanmean(np.abs(right - left)))
                    if left is not None and right is not None
                    else np.nan
                )
            array = np.asarray(values, dtype=np.float64)
            if np.isfinite(array).any():
                series.append({"x": drift_x, "y": array, "label": label})
                n_samples += int(np.isfinite(array).sum())
        panels.append(
            _Panel(
                kind="line",
                title="P / 策略漂移",
                xlabel="Episode e（相对 e-1）",
                ylabel="mean |Δ|",
                options={"series": series},
            )
        )
    else:
        panels.append(_Panel(kind="text", title="价值/策略漂移", options={"text": "episode 数不足（<2）"}))

    macro = blocks["macro"]
    if macro:
        keys = list(macro)
        x = np.arange(len(keys), dtype=np.float64)
        series = []
        for metric, label in (
            ("fc1_one_step_hatc_mae", "hatc 一步 MAE"),
            ("fc1_one_step_lnk_mae", "lnk 一步 MAE"),
            ("sdf_g_max", "SDF 矩约束最大违反"),
        ):
            array = np.asarray(
                [float(_result_metrics(macro[key]).get(metric, float("nan"))) for key in keys],
                dtype=np.float64,
            )
            if np.isfinite(array).any():
                series.append({"x": x, "y": array, "label": label})
                n_samples += int(np.isfinite(array).sum())
        panels.append(
            _Panel(
                kind="line",
                title="FC1 一步误差与 SDF 约束",
                xlabel="Episode",
                ylabel="指标值",
                options={"series": series},
            )
        )
    else:
        panels.append(_Panel(kind="text", title="宏观 / SDF", options={"text": "无 macro 结果"}))

    sim = blocks["sim"]
    if sim:
        keys = list(sim)
        x = np.arange(len(keys), dtype=np.float64)
        series = []
        for metric, label in (
            ("sim_baseline_capital_change_t0_t1", "t0→t1 资本变化"),
            ("sim_baseline_initial_cohort_default_share_t1", "初始 cohort 违约份额"),
        ):
            array = np.asarray(
                [float(_result_metrics(sim[key]).get(metric, float("nan"))) for key in keys],
                dtype=np.float64,
            )
            if np.isfinite(array).any():
                series.append({"x": x, "y": array, "label": label})
                n_samples += int(np.isfinite(array).sum())
        panels.append(
            _Panel(
                kind="line",
                title="固定冲击 rollout 汇总",
                xlabel="Episode",
                ylabel="指标值",
                options={"series": series},
            )
        )
    else:
        panels.append(_Panel(kind="text", title="模拟 / 分布", options={"text": "无 sim 结果"}))

    coverage = "结果块覆盖：" + ", ".join(
        f"{name}={'有' if blocks[name] else '无'}" for name in blocks
    )
    panels.append(_Panel(kind="text", title="数据覆盖", options={"text": coverage}))
    panels.append(
        _Panel(
            kind="text",
            title="判据说明",
            options={
                "text": (
                    "收敛 = 企业值/策略 + 宏观信念 + 定价核 同时稳定\n"
                    "训练 loss 不作收敛证据\n"
                    "阈值未配置时不判 PASS/FAIL（THRESHOLD_NOT_CONFIGURED）"
                )
            },
        )
    )

    meta = _base_meta(
        episode="多 episode 汇总",
        eta="η∈{0,1}",
        i_slice="bank 全体",
        data_source="多 bank 汇总",
        capital_weighted="各面板见自身口径",
        n_samples=n_samples,
        extra={"result_blocks": present},
    )
    return _render_grid(
        "F12",
        panels,
        rows=2,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F12"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode="多 episode", eta="η∈{0,1}",
            data_source="多 bank 汇总", capital_weighted="混合口径", n_samples=n_samples,
        ),
    )


# ---------------------------------------------------------------------------
# V-series renderers
# ---------------------------------------------------------------------------


def _value_series_figure(
    figure_id: str,
    surface_key: str,
    family: str,
    *,
    policy_results,
    output_dir,
    style=None,
    episode=None,
    annotations=None,
) -> FigureRecord:
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip(figure_id, reason or REASON_NO_POLICY_SURFACES)
    axes_pair = grid_axes(result)
    if axes_pair is None:
        return _skip(figure_id, REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    if (
        _surface(result, f"eta1_{surface_key}") is None
        and _surface(result, f"eta0_{surface_key}") is None
    ):
        return _skip(figure_id, REASON_NO_POLICY_SURFACES)
    style = resolve_style(style, [result])
    region = _default_region(result, (b_values.size, z_values.size))
    overlays = _boundary_overlay(result, style)

    panels = []
    for view in ("eta1", "eta0"):
        view_values = _surface(result, f"{view}_{surface_key}")
        if view_values is None:
            continue
        panels.append(
            _Panel(
                kind="3d",
                title=f"3D {surface_key}（{_eta_label(view)}）",
                family=family,
                values=view_values,
                options={"b_values": b_values, "z_values": z_values},
            )
        )
        panels.append(
            _Panel(
                kind="contour",
                title=f"等高线 {surface_key}（{_eta_label(view)}）+ 违约边界",
                family=family,
                values=view_values,
                options={"b_values": b_values, "z_values": z_values,
                         "region": region, "overlays": overlays},
            )
        )
    while len(panels) < 4:
        panels.append(_Panel(kind="text", title="（无数据）", options={"text": "该 η 视图曲面缺失"}))

    n_samples = int(sum(_n_finite(item.values) for item in panels if item.values is not None))
    meta = _base_meta(
        episode=episode_key,
        eta="η=1 / η=0 双视图",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
        extra={"surface": surface_key, "canonical_3d_note": "canonical 集见 F01"},
    )
    return _render_grid(
        figure_id,
        panels,
        rows=2,
        cols=2,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS[figure_id],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=episode_key, eta="η=1/η=0", i_slice=_i_slice_text(result),
            data_source=str(meta["data_source"]), capital_weighted="否", n_samples=n_samples,
        ),
    )


def render_v01(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """V01: P0 surface."""
    return _value_series_figure(
        "V01", "P0", "value", policy_results=policy_results,
        output_dir=output_dir, style=style, episode=episode, annotations=annotations,
    )


def render_v02(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """V02: PI surface with the investment cost ``i`` annotated."""
    return _value_series_figure(
        "V02", "PI", "value", policy_results=policy_results,
        output_dir=output_dir, style=style, episode=episode, annotations=annotations,
    )


def render_v03(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """V03: equity P with Phat and the Phat<=0 default region."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("V03", reason or REASON_NO_POLICY_SURFACES)
    axes_pair = grid_axes(result)
    if axes_pair is None:
        return _skip("V03", REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    p_values = _surface(result, "eta1_P")
    phat = _surface(result, "eta1_Phat")
    if p_values is None or phat is None:
        return _skip("V03", REASON_NO_POLICY_SURFACES)
    style = resolve_style(style, [result])
    region = phat <= 0.0
    overlays = _boundary_overlay(result, style)
    panels = [
        _Panel(kind="3d", title="3D P（η=1，投资选择后股权价值）", family="value", values=p_values,
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="contour", title="P + Phat≤0 违约区", family="value", values=p_values,
               options={"b_values": b_values, "z_values": z_values,
                        "region": region, "overlays": overlays}),
        _Panel(kind="heatmap", title="Phat(b,z)（硬违约判别，≤0 违约）", family="value", values=phat,
               options={"b_values": b_values, "z_values": z_values,
                        "region": region, "overlays": overlays}),
        _Panel(kind="heatmap", title="违约区指示（Phat≤0）", family="prob",
               values=region.astype(np.float64),
               options={"b_values": b_values, "z_values": z_values, "overlays": overlays}),
    ]
    n_samples = int(_n_finite(p_values))
    meta = _base_meta(
        episode=episode_key,
        eta="η=1",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
        extra={"default_region_share": float(region.mean()),
               "phat_semantics": "Phat≤0 为硬违约；bar_z 是软概率，不混用"},
    )
    return _render_grid(
        "V03", panels, rows=2, cols=2, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["V03"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=episode_key, eta="η=1", i_slice=_i_slice_text(result),
            data_source=str(meta["data_source"]), capital_weighted="否", n_samples=n_samples,
        ),
    )


def render_v04(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """V04: the four Q calibers (claim / unit / effective / recovery)."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("V04", reason or REASON_NO_POLICY_SURFACES)
    axes_pair = grid_axes(result)
    if axes_pair is None:
        return _skip("V04", REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    spec = (
        ("eta1_Q_claim", "Q Claim = b₊·q_unit（发行定价口径）"),
        ("eta1_q_unit", "q_unit（单位债券价）"),
        ("eta1_Q_effective", "Q Effective（结算口径，hybrid 含违约硬门）"),
        ("eta1_recovery", "recovery = φ(1-δ+exp(x+z))（违约回收）"),
    )
    panels = []
    for key, label in spec:
        values = _surface(result, key)
        if values is None:
            return _skip("V04", REASON_NO_Q_SURFACES)
        panels.append(
            _Panel(
                kind="heatmap",
                title=label,
                family="q",
                values=values,
                options={"b_values": b_values, "z_values": z_values,
                         "region": _default_region(result, values.shape),
                         "overlays": _boundary_overlay(result, style)},
            )
        )
    style = resolve_style(style, [result])
    n_samples = int(sum(_n_finite(item.values) for item in panels))
    meta = _base_meta(
        episode=episode_key,
        eta="η=1",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
        extra={"q_calibers": ["Q_claim", "q_unit", "Q_effective", "recovery"]},
    )
    return _render_grid(
        "V04", panels, rows=2, cols=2, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["V04"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=episode_key, eta="η=1", i_slice=_i_slice_text(result),
            data_source=str(meta["data_source"]), capital_weighted="否", n_samples=n_samples,
        ),
    )


def render_v05(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """V05: value contours + default boundaries."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("V05", reason or REASON_NO_POLICY_SURFACES)
    axes_pair = grid_axes(result)
    if axes_pair is None:
        return _skip("V05", REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    style = resolve_style(style, [result])
    overlays = _boundary_overlay(result, style)
    region = _default_region(result, (b_values.size, z_values.size))
    panels = []
    for key, label in (("eta1_P0", "P0"), ("eta1_PI", "PI"), ("eta1_P", "P")):
        values = _surface(result, key)
        if values is None:
            return _skip("V05", REASON_NO_POLICY_SURFACES)
        panels.append(
            _Panel(
                kind="contour",
                title=f"{label} 等高线 + 硬/软违约边界",
                family=_family_of(key),
                values=values,
                options={"b_values": b_values, "z_values": z_values,
                         "region": region, "overlays": overlays},
            )
        )
    hard_soft = _pick_table(result, "hard_soft_boundary")
    if hard_soft is not None and {"b", "z_hard_phat0"}.issubset(hard_soft.columns):
        panels.append(
            _Panel(
                kind="line",
                title="硬边界 Phat=0 vs 软边界 bar_z=0.5",
                xlabel="b = 杠杆率",
                ylabel="z 边界",
                options={
                    "series": [
                        {"x": _table_column(hard_soft, "b"),
                         "y": _table_column(hard_soft, "z_hard_phat0"), "label": "Phat=0"},
                        {"x": _table_column(hard_soft, "b"),
                         "y": _table_column(hard_soft, "z_soft_barz0p5"),
                         "label": "bar_z=0.5", "linestyle": "--"},
                    ]
                },
            )
        )
    else:
        panels.append(_Panel(kind="text", title="边界对照", options={"text": "hard_soft_boundary 表缺失"}))
    n_samples = int(sum(_n_finite(item.values) for item in panels if item.values is not None))
    meta = _base_meta(
        episode=episode_key,
        eta="η=1",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
    )
    return _render_grid(
        "V05", panels, rows=2, cols=2, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["V05"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=episode_key, eta="η=1", i_slice=_i_slice_text(result),
            data_source=str(meta["data_source"]), capital_weighted="否", n_samples=n_samples,
        ),
    )


def render_v06(*, policy_results, output_dir, style=None, annotations=None) -> FigureRecord:
    """V06: value drift across episodes."""
    episodes = _as_episode_map(policy_results)
    if len(episodes) < 2:
        return _skip("V06", REASON_INSUFFICIENT_EPISODES)
    keys = list(episodes)
    axes_pair = grid_axes(episodes[keys[-1]])
    if axes_pair is None:
        return _skip("V06", REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    style = resolve_style(style, list(episodes.values()))

    panels = []
    n_samples = 0
    summary_series = []
    for key, family in (("eta1_P0", "P0"), ("eta1_PI", "PI"), ("eta1_P", "P")):
        differences = []
        means = []
        for previous, current in zip(keys[:-1], keys[1:]):
            left = _surface(episodes[previous], key)
            right = _surface(episodes[current], key)
            if left is None or right is None or left.shape != right.shape:
                differences.append(None)
                means.append(np.nan)
                continue
            delta = right - left
            differences.append(delta)
            good = _finite(delta)
            means.append(float(np.mean(np.abs(good))) if good.size else np.nan)
        array = np.asarray(means, dtype=np.float64)
        if np.isfinite(array).any():
            summary_series.append(
                {"x": np.arange(1, len(keys), dtype=np.float64), "y": array,
                 "label": f"mean|Δ{family}|"}
            )
        last_delta = next((item for item in reversed(differences) if item is not None), None)
        if last_delta is not None:
            panels.append(
                _Panel(
                    kind="heatmap",
                    title=f"Δ{family}（{keys[-2]} → {keys[-1]}）",
                    family="residual",
                    values=last_delta,
                    options={"b_values": b_values, "z_values": z_values,
                             "region": _default_region(episodes[keys[-1]], last_delta.shape)},
                )
            )
            n_samples += _n_finite(last_delta)
    if not panels:
        return _skip("V06", REASON_NO_POLICY_SURFACES)
    while len(panels) < 3:
        panels.append(_Panel(kind="text", title="（无数据）", options={"text": "该曲面缺失"}))
    panels.append(
        _Panel(
            kind="line",
            title="跨集价值漂移汇总",
            xlabel="Episode e（相对 e-1）",
            ylabel="mean |Δvalue|",
            options={"series": summary_series},
        )
    )
    meta = _base_meta(
        episode=f"{keys[0]}..{keys[-1]}",
        eta="η=1",
        i_slice=_i_slice_text(episodes[keys[-1]]),
        data_source=_describe_bank_source(getattr(episodes[keys[-1]], "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
        extra={"episodes": [key for key in keys]},
    )
    return _render_grid(
        "V06", panels, rows=2, cols=2, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["V06"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=meta["episode"], eta="η=1", i_slice=meta["i_slice"],
            data_source=str(meta["data_source"]), capital_weighted="否", n_samples=n_samples,
        ),
    )


# ---------------------------------------------------------------------------
# P-series renderers
# ---------------------------------------------------------------------------


def render_p01(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """P01: Grid bp0* surface."""
    return _value_series_figure(
        "P01", "bp0", "bp", policy_results=policy_results,
        output_dir=output_dir, style=style, episode=episode, annotations=annotations,
    )


def render_p02(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """P02: Grid bpI* surface."""
    return _value_series_figure(
        "P02", "bpI", "bp", policy_results=policy_results,
        output_dir=output_dir, style=style, episode=episode, annotations=annotations,
    )


def render_p03(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """P03: effective debt policy (executed ``b_next``)."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("P03", reason or REASON_NO_POLICY_ROWS)
    rows = _pick_table(result, "policy_rows")
    required = {"b", "z", "eta", "b_next_mix", "bp_star_mix"}
    if rows is None or not required.issubset(rows.columns):
        return _skip("P03", REASON_NO_POLICY_ROWS)
    style = resolve_style(style, [result])

    b = _table_column(rows, "b")
    z = _table_column(rows, "z")
    eta = _table_column(rows, "eta")
    b_next = _table_column(rows, "b_next_mix")
    bp_star = _table_column(rows, "bp_star_mix")
    eta1 = eta > 0.5
    eta0 = ~eta1

    head_values = None
    if "bp0_head" in rows.columns:
        head_values = _table_column(rows, "bp0_head")
    panels = [
        _Panel(
            kind="scatter",
            title="b_next vs b（η=0 强制 b_next=b）",
            xlabel="b 当前杠杆",
            ylabel="b_next = η·bp* + (1-η)·b",
            options={
                "series": [
                    {"x": b[eta0], "y": b_next[eta0], "label": "η=0（refi 关闭）", "color": "tab:blue"},
                    {"x": b[eta1], "y": b_next[eta1], "label": "η=1（refi active）", "color": "tab:orange"},
                ]
            },
        ),
        _Panel(
            kind="scatter",
            title="b_next vs z（生产率对执行杠杆的影响）",
            xlabel="z = 企业生产率冲击",
            ylabel="b_next",
            options={
                "series": [
                    {"x": z[eta0], "y": b_next[eta0], "label": "η=0", "color": "tab:blue"},
                    {"x": z[eta1], "y": b_next[eta1], "label": "η=1", "color": "tab:orange"},
                ],
                "diag": False,
            },
        ),
        _Panel(
            kind="scatter",
            title="Grid bp* vs 头输出 bp_head（头仅兼容）",
            xlabel="Grid bp*（正式策略）",
            ylabel="bp_head（兼容输出）",
            options={
                "series": [
                    {"x": bp_star[eta1],
                     "y": (head_values[eta1] if head_values is not None else bp_star[eta1]),
                     "label": "η=1"},
                ]
            },
        ),
    ]
    delta = b_next - b
    panels.append(
        _Panel(
            kind="line",
            title="Δb = b_next - b（按 η）",
            xlabel="b 当前杠杆",
            ylabel="Δb",
            options={
                "series": [
                    {"x": b[eta0], "y": delta[eta0], "label": "η=0（应恒为 0）",
                     "marker": "o", "markersize": 2},
                    {"x": b[eta1], "y": delta[eta1], "label": "η=1",
                     "marker": "o", "markersize": 2},
                ]
            },
        )
    )

    n_samples = int(len(rows))
    eta0_error = float(np.nanmax(np.abs(delta[eta0]))) if eta0.any() else float("nan")
    meta = _base_meta(
        episode=episode_key,
        eta="η=0 / η=1",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
        extra={
            "financing_gate": "b_next = eta_t * bp + (1 - eta_t) * b_t",
            "eta0_financing_gate_max_error": eta0_error,
            "formal_policy": "grid_bp (BPGridTeacher)",
        },
    )
    return _render_grid(
        "P03", panels, rows=2, cols=2, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["P03"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=episode_key, eta="η=0/η=1", i_slice=meta["i_slice"],
            data_source=str(meta["data_source"]), capital_weighted="否", n_samples=n_samples,
        ),
    )


def render_p04(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """P04: Bellman objective vs candidate ``bp`` with the Q claim overlay."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("P04", reason or REASON_NO_OBJECTIVE_SLICES)
    slices = _pick_table(result, "objective_slices")
    required = {"row_index", "branch", "bp_candidate", "bellman_rhs", "q_claim_issue"}
    if slices is None or slices.empty or not required.issubset(slices.columns):
        return _skip("P04", REASON_NO_OBJECTIVE_SLICES)
    style = resolve_style(style, [result])

    groups = list(slices.groupby(["row_index", "branch"], sort=True))
    panels = []
    n_samples = 0
    for (row_index, branch), group in groups[:4]:
        bp = _table_column(group, "bp_candidate")
        rhs = _table_column(group, "bellman_rhs")
        q_claim = _table_column(group, "q_claim_issue")
        cashflow = _table_column(group, "cashflow") if "cashflow" in group.columns else None
        continuation = _table_column(group, "continuation") if "continuation" in group.columns else None
        series = [{"x": bp, "y": rhs, "label": "Bellman RHS"}]
        if cashflow is not None:
            series.append({"x": bp, "y": cashflow, "label": "cashflow", "linestyle": "--"})
        if continuation is not None:
            series.append({"x": bp, "y": continuation, "label": "continuation", "linestyle": ":"})
        if q_claim is not None:
            series.append({"x": bp, "y": q_claim, "label": "Q_claim（发行定价）", "linestyle": "-."})
        eta_value = float(group["eta"].iloc[0]) if "eta" in group.columns else float("nan")
        panels.append(
            _Panel(
                kind="line",
                title=f"row {int(row_index)}｜branch={branch}｜η={eta_value:g}",
                xlabel="候选下一期杠杆 bp",
                ylabel="目标值 / 发行定价",
                options={"series": series},
            )
        )
        n_samples += int(len(group))
    while len(panels) < 4:
        panels.append(_Panel(kind="text", title="（无数据）", options={"text": "代表性状态不足"}))

    metrics = _result_metrics(result)
    flatness = float(metrics.get("objective_slice_eta0_flatness_max_abs", float("nan")))
    meta = _base_meta(
        episode=episode_key,
        eta="η=0 / η=1（η=0 目标应为常数）",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
        extra={"eta0_objective_flatness_max_abs": flatness,
               "representative_states": sorted({int(v) for v in slices["row_index"].tolist()})},
    )
    return _render_grid(
        "P04", panels, rows=2, cols=2, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["P04"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=episode_key, eta="η=0/η=1", i_slice=meta["i_slice"],
            data_source=str(meta["data_source"]), capital_weighted="否", n_samples=n_samples,
        ),
    )


# ---------------------------------------------------------------------------
# M-series renderers
# ---------------------------------------------------------------------------


def _regression_stats(calculated, forecast) -> str:
    mask = np.isfinite(calculated) & np.isfinite(forecast)
    if not mask.any():
        return "n=0"
    err = forecast[mask] - calculated[mask]
    denominator = float(np.sum((calculated[mask] - np.mean(calculated[mask])) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / denominator if denominator > 0 else float("nan")
    return (
        f"n={int(mask.sum())}  MAE={np.mean(np.abs(err)):.4g}  "
        f"Bias={np.mean(err):.4g}  R2={r2:.4g}"
    )


def render_m01(*, macro_results, output_dir, style=None, annotations=None) -> FigureRecord:
    """M01: FC1 one-step scatter."""
    episodes = _as_episode_map(macro_results)
    if not episodes:
        return _skip("M01", REASON_NO_MACRO_PAIRS)
    key = list(episodes)[-1]
    result = episodes[key]
    frame = _macro_pairs(result)
    if frame is None:
        return _skip("M01", REASON_NO_MACRO_PAIRS)
    style = resolve_style(style, [result])

    hatc_x = _table_column(frame, "hatc_t")
    hatc_y = _table_column(frame, "hatcf_t")
    lnk_x = _table_column(frame, "lnk_t")
    lnk_y = _table_column(frame, "lnkf_t")
    panels = [
        _Panel(
            kind="scatter",
            title="Hatcf_t vs Hatc_t",
            xlabel="Hatc_t 实现（log(C/K)）",
            ylabel="Hatcf_t 信念预测",
            options={"series": [{"x": hatc_x, "y": hatc_y, "label": "hatc"}]},
        ),
        _Panel(
            kind="scatter",
            title="LnKF_t vs LnK_t",
            xlabel="LnK_t 实现（log ΣK）",
            ylabel="LnKF_t 信念预测",
            options={"series": [{"x": lnk_x, "y": lnk_y, "label": "lnk"}]},
        ),
        _Panel(
            kind="text",
            title="一步预测统计",
            options={
                "text": (
                    "hatc: " + _regression_stats(hatc_x, hatc_y) + "\n"
                    "lnk:  " + _regression_stats(lnk_x, lnk_y)
                )
            },
        ),
        _Panel(
            kind="text",
            title="对齐语义",
            options={
                "text": (
                    "hatcf_t/lnkf_t = t-1 对节点 t 的预测\n"
                    "hatc_t/lnk_t = 节点 t 实现聚合\n"
                    "同行对齐 = shift=0（一步预测）\n"
                    "配对键 (path,t,branch,economic_node_id)"
                )
            },
        ),
    ]
    n_samples = int(len(frame))
    meta = _base_meta(
        episode=key,
        eta="不适用",
        i_slice="不适用",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
    )
    return _render_grid(
        "M01", panels, rows=2, cols=2, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["M01"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=key, data_source=str(meta["data_source"]),
            capital_weighted="否", n_samples=n_samples,
        ),
    )


def _path_figure(
    figure_id: str,
    *,
    macro_results,
    output_dir,
    style=None,
    annotations=None,
    realized: str,
    forecast: str,
    ylabel: str,
    title: str,
) -> FigureRecord:
    episodes = _as_episode_map(macro_results)
    if not episodes:
        return _skip(figure_id, REASON_NO_MACRO_PAIRS)
    key = list(episodes)[-1]
    result = episodes[key]
    frame = _macro_pairs(result)
    if frame is None or not {"path", "t", realized, forecast}.issubset(frame.columns):
        return _skip(figure_id, REASON_NO_MACRO_PAIRS)
    style = resolve_style(style, [result])

    path_column = pd.to_numeric(frame["path"], errors="coerce")
    paths = sorted(path_column.dropna().unique())
    selected = paths[:6]
    panels = []
    n_samples = 0
    for path_id in selected:
        subset = frame.loc[path_column == path_id].sort_values("t", kind="mergesort")
        panels.append(
            _Panel(
                kind="line",
                title=f"{title}｜path {int(path_id)}",
                xlabel="t（宏观节点）",
                ylabel=ylabel,
                options={
                    "series": [
                        {"x": _table_column(subset, "t"), "y": _table_column(subset, forecast),
                         "label": forecast, "linestyle": "--"},
                        {"x": _table_column(subset, "t"), "y": _table_column(subset, realized),
                         "label": realized},
                    ]
                },
            )
        )
        n_samples += int(len(subset))
    if not panels:
        return _skip(figure_id, REASON_NO_MACRO_PAIRS)
    while len(panels) < 6:
        panels.append(_Panel(kind="text", title="（无数据）", options={"text": "path 数量不足"}))

    meta = _base_meta(
        episode=key,
        eta="不适用",
        i_slice="不适用",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
        extra={"paths": [int(value) for value in selected], "n_paths_total": int(len(paths))},
    )
    return _render_grid(
        figure_id, panels, rows=2, cols=3, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS[figure_id], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=key, data_source=str(meta["data_source"]),
            capital_weighted="否", n_samples=n_samples,
        ),
    )


def render_m02(*, macro_results, output_dir, style=None, annotations=None) -> FigureRecord:
    """M02: Hatcf vs Hatc paths."""
    return _path_figure(
        "M02", macro_results=macro_results, output_dir=output_dir, style=style,
        annotations=annotations, realized="hatc_t", forecast="hatcf_t",
        ylabel="Hatc / Hatcf", title="Hatcf vs Hatc",
    )


def render_m03(*, macro_results, output_dir, style=None, annotations=None) -> FigureRecord:
    """M03: LnKF vs LnK paths."""
    return _path_figure(
        "M03", macro_results=macro_results, output_dir=output_dir, style=style,
        annotations=annotations, realized="lnk_t", forecast="lnkf_t",
        ylabel="LnK / LnKF", title="LnKF vs LnK",
    )


def render_m04(*, macro_results, output_dir, style=None, annotations=None) -> FigureRecord:
    """M04: macro recursive forecast error."""
    episodes = _as_episode_map(macro_results)
    if not episodes:
        return _skip("M04", REASON_NO_RECURSION_TABLE)
    key = list(episodes)[-1]
    result = episodes[key]
    table = _pick_table(result, "fc1_recursion_errors")
    required = {"caliber", "variable", "horizon", "mae", "rmse", "bias"}
    if table is None or table.empty or not required.issubset(table.columns):
        return _skip("M04", REASON_NO_RECURSION_TABLE)
    style = resolve_style(style, [result])

    panels = []
    for variable, label in (("hatc", "hatc"), ("lnk", "lnk")):
        subset = table.loc[table["variable"].astype(str) == variable]
        series_mae = []
        series_bias = []
        for caliber, group in subset.groupby("caliber"):
            horizon = _table_column(group, "horizon")
            series_mae.append({"x": horizon, "y": _table_column(group, "mae"),
                               "label": f"{caliber} MAE"})
            series_mae.append({"x": horizon, "y": _table_column(group, "rmse"),
                               "label": f"{caliber} RMSE", "linestyle": "--"})
            series_bias.append({"x": horizon, "y": _table_column(group, "bias"),
                                "label": f"{caliber} bias"})
        panels.append(
            _Panel(
                kind="line",
                title=f"{label} 递归误差 vs horizon",
                xlabel="horizon h（递归步数）",
                ylabel="误差（MAE / RMSE）",
                options={"series": series_mae},
            )
        )
        panels.append(
            _Panel(
                kind="line",
                title=f"{label} 递归偏差 vs horizon",
                xlabel="horizon h",
                ylabel="bias（forecast - realized）",
                options={"series": series_bias},
            )
        )
    metrics = _result_metrics(result)
    lines = [
        f"{name} = {float(metrics[name]):.6g}"
        for name in sorted(metrics)
        if "drift_slope" in name or "out_of_training_range" in name or name.endswith("_diverged")
    ]
    panels.append(_Panel(kind="text", title="漂移与出训练范围",
                         options={"text": "\n".join(lines[:18]) or "无"}))
    panels.append(
        _Panel(
            kind="text",
            title="两口径定义",
            options={
                "text": (
                    "production_one_step：每次 FC1 喂父节点算得宏观态\n"
                    "autonomous_recursive：完全自回归，误差随 h 累积\n"
                    "两口径分开报，不合并"
                )
            },
        )
    )
    while len(panels) < 6:
        panels.append(_Panel(kind="text", title="（无数据）", options={"text": "该变量缺失"}))

    n_samples = int(table["n"].sum()) if "n" in table.columns else int(len(table))
    meta = _base_meta(
        episode=key,
        eta="不适用",
        i_slice="不适用",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
        extra={"calibers": sorted(set(table["caliber"].astype(str)))},
    )
    return _render_grid(
        "M04", panels, rows=2, cols=3, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["M04"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=key, data_source=str(meta["data_source"]),
            capital_weighted="否", n_samples=n_samples,
        ),
    )


def render_m05(*, macro_results, output_dir, style=None, annotations=None) -> FigureRecord:
    """M05: SDF distribution and constraints."""
    episodes = _as_episode_map(macro_results)
    if not episodes:
        return _skip("M05", REASON_NO_SDF_TABLE)
    key = list(episodes)[-1]
    result = episodes[key]
    table = _pick_table(result, "sdf_macro_state_summary")
    metrics = _result_metrics(result)
    sdf_metrics = {name: value for name, value in metrics.items() if str(name).startswith("sdf_")}
    if table is None and not sdf_metrics:
        return _skip("M05", REASON_NO_SDF_TABLE)
    style = resolve_style(style, [result])

    panels = []
    if table is not None and not table.empty:
        labels = table["macro_state"].astype(str).tolist()
        panels.append(
            _Panel(kind="bar", title="E[M | 宏观态]（± 标准误）",
                   xlabel="宏观态", ylabel="M 均值",
                   options={"labels": labels, "values": _table_column(table, "M_mean"),
                            "yerr": _table_column(table, "M_mean_se"), "rotation": 15})
        )
        panels.append(
            _Panel(kind="bar", title="归一化财富 Euler 残差（± 标准误）",
                   xlabel="宏观态", ylabel="mean residual",
                   options={"labels": labels, "values": _table_column(table, "mean_residual"),
                            "yerr": _table_column(table, "se_residual"), "rotation": 15})
        )
        panels.append(
            _Panel(kind="bar", title="各宏观态样本量",
                   xlabel="宏观态", ylabel="n_parents",
                   options={"labels": labels, "values": _table_column(table, "n_parents"),
                            "rotation": 15})
        )
    else:
        for title in ("宏观态分解", "残差分解", "样本量"):
            panels.append(_Panel(kind="text", title=title,
                                 options={"text": "sdf_macro_state_summary 缺失"}))

    labels = []
    values = []
    for name in ("sdf_g_mean_low", "sdf_g_mean_high", "sdf_g_var_high", "sdf_g_max"):
        if name in sdf_metrics:
            labels.append(name.replace("sdf_", ""))
            values.append(float(sdf_metrics[name]))
    panels.append(
        _Panel(kind="bar", title="矩约束违反量（g_*，应为 ≤0）",
               xlabel="约束", ylabel="违反量",
               options={"labels": labels, "values": np.asarray(values, dtype=np.float64),
                        "rotation": 15})
    )
    lines = []
    for name in (
        "sdf_M_mean", "sdf_M_mean_se", "sdf_M_var", "sdf_M_std",
        "sdf_signed_aio_estimate", "sdf_signed_aio_se", "sdf_signed_aio_negative_share",
        "sdf_population_target_estimate", "sdf_population_target_se",
        "sdf_population_target_t_stat", "sdf_nonfinite_ratio",
        "sdf_log_R_clip_share", "sdf_valid_parent_ratio",
    ):
        if name in sdf_metrics:
            lines.append(f"{name} = {float(sdf_metrics[name]):.6g}")
    panels.append(_Panel(kind="text", title="SDF 分布与数值安全",
                         options={"text": "\n".join(lines) or "无"}))
    panels.append(
        _Panel(
            kind="text",
            title="解读护栏",
            options={
                "text": (
                    "Signed AiO 负样本均值 ≠ 负总体条件矩\n"
                    "总体目标另报 sdf_population_target_*\n"
                    "约束违反量与 AiO 分开报告\n"
                    "M 口径：rawM vs trainM(clipped) 须声明"
                )
            },
        )
    )
    while len(panels) < 6:
        panels.append(_Panel(kind="text", title="（无数据）", options={"text": "该面板数据缺失"}))

    n_samples = (
        int(table["n_parents"].sum())
        if table is not None and "n_parents" in table.columns
        else len(sdf_metrics)
    )
    meta = _base_meta(
        episode=key,
        eta="不适用",
        i_slice="不适用",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="否",
        n_samples=n_samples,
        extra={"sdf_metric_count": len(sdf_metrics)},
    )
    return _render_grid(
        "M05", panels, rows=2, cols=3, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["M05"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=key, data_source=str(meta["data_source"]),
            capital_weighted="否", n_samples=n_samples,
        ),
    )


def render_m06(*, sim_results, output_dir, style=None, annotations=None) -> FigureRecord:
    """M06: C/K, K, C, Y, I over time."""
    episodes = _as_episode_map(sim_results)
    if not episodes:
        return _skip("M06", REASON_NO_MACRO_PANEL)
    key = list(episodes)[-1]
    result = episodes[key]
    macro_panel = _pick_table(result, "macro_panel")
    if macro_panel is None or macro_panel.empty or not {"t", "K", "C"}.issubset(macro_panel.columns):
        return _skip("M06", REASON_NO_MACRO_PANEL)
    style = resolve_style(style, [result])

    frame = macro_panel
    if {"path", "branch"}.issubset(frame.columns):
        branch = pd.to_numeric(frame["branch"], errors="coerce")
        parent = frame.loc[branch < 0]
        if not parent.empty:
            frame = parent
        else:
            child = frame.loc[branch == 0]
            if not child.empty:
                frame = child
    frame = frame.sort_values("t", kind="mergesort")

    def _aggregate(values):
        if values is None:
            return None, None
        grouped = pd.DataFrame({"t": _table_column(frame, "t"), "v": values}).groupby(
            "t", sort=True
        )["v"].mean()
        return grouped.index.to_numpy(dtype=np.float64), grouped.to_numpy(dtype=np.float64)

    t_k, k_series = _aggregate(_table_column(frame, "K"))
    t_c, c_series = _aggregate(_table_column(frame, "C"))
    t_h, hatc_series = _aggregate(_table_column(frame, "Hatc"))
    ck_series = np.exp(hatc_series) if hatc_series is not None else None

    panels = []
    if ck_series is not None and t_h is not None:
        panels.append(
            _Panel(kind="line", title="C/K（= exp(Hatc)，Hatc=log(C/K)）",
                   xlabel="t（模拟期）", ylabel="C/K",
                   options={"series": [{"x": t_h, "y": ck_series, "label": "C/K"}]})
        )
    if k_series is not None:
        panels.append(
            _Panel(kind="line", title="总资本 K", xlabel="t", ylabel="K = Σ K_i",
                   options={"series": [{"x": t_k, "y": k_series, "label": "K"}]})
        )
    if c_series is not None:
        panels.append(
            _Panel(kind="line", title="总消费 C", xlabel="t", ylabel="C",
                   options={"series": [{"x": t_c, "y": c_series, "label": "C"}]})
        )

    firm_panel = _sim_firm_panel(result)
    if firm_panel is not None and {"t", "Y"}.issubset(firm_panel.columns):
        grouped = pd.DataFrame(
            {"t": _table_column(firm_panel, "t"), "v": _table_column(firm_panel, "Y")}
        ).groupby("t", sort=True)["v"].sum()
        panels.append(
            _Panel(kind="line", title="总产出 Y", xlabel="t", ylabel="Y = Σ Y_i",
                   options={"series": [{"x": grouped.index.to_numpy(dtype=np.float64),
                                        "y": grouped.to_numpy(dtype=np.float64), "label": "Y"}]})
        )
    else:
        panels.append(_Panel(kind="text", title="总产出 Y",
                             options={"text": "firm_panel 无 Y 列（本面板省略）"}))

    drawn_investment = False
    for column, label in (("I_total", "总投资 I"), ("I_oper", "运营投资"), ("I_entry", "进入投资")):
        if column in frame.columns:
            t_i, values = _aggregate(_table_column(frame, column))
            if values is not None and t_i is not None:
                panels.append(
                    _Panel(kind="line", title=label, xlabel="t", ylabel=column,
                           options={"series": [{"x": t_i, "y": values, "label": column}]})
                )
                drawn_investment = True
                break
    if not drawn_investment:
        panels.append(_Panel(kind="text", title="投资 I",
                             options={"text": "macro_panel 无 I_* 列（本面板省略）"}))

    while len(panels) < 6:
        panels.append(_Panel(kind="text", title="（无数据）", options={"text": "该聚合量缺失"}))

    n_samples = int(len(frame))
    meta = _base_meta(
        episode=key,
        eta="不适用（模拟聚合）",
        i_slice="不适用",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="总量口径（Σ）",
        extinct_paths=_describe_extinct(result),
        n_samples=n_samples,
        extra={"aggregation": "同 (path,t) 内跨企业求和/均值，path 间再平均",
               "view_filter": "branch<0 优先（parent 视图），否则 branch==0"},
    )
    return _render_grid(
        "M06", panels, rows=2, cols=3, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["M06"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=key, data_source=str(meta["data_source"]),
            capital_weighted="总量口径", extinct_paths=str(meta["extinct_paths"]),
            n_samples=n_samples,
        ),
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

_SINGLE_RENDERERS = (
    ("F01", render_f01, ("policy_results",)),
    ("F02", render_f02, ("policy_results",)),
    ("F03", render_f03, ("policy_results",)),
    ("F04", render_f04, ("policy_results",)),
    ("F05", render_f05, ("bellman_results",)),
    ("F06", render_f06, ("policy_results", "comparison")),
    ("F07", render_f07, ("macro_results",)),
    ("F08", render_f08, ("macro_results",)),
    ("F09", render_f09, ("sim_results",)),
    ("F10", render_f10, ("sim_results",)),
    ("F11", render_f11_waterfall, ("sim_results",)),
    ("F12", render_f12,
     ("bellman_results", "policy_results", "macro_results", "sim_results", "comparison")),
    ("V01", render_v01, ("policy_results",)),
    ("V02", render_v02, ("policy_results",)),
    ("V03", render_v03, ("policy_results",)),
    ("V04", render_v04, ("policy_results",)),
    ("V05", render_v05, ("policy_results",)),
    ("V06", render_v06, ("policy_results",)),
    ("P01", render_p01, ("policy_results",)),
    ("P02", render_p02, ("policy_results",)),
    ("P03", render_p03, ("policy_results",)),
    ("P04", render_p04, ("policy_results",)),
    ("M01", render_m01, ("macro_results",)),
    ("M02", render_m02, ("macro_results",)),
    ("M03", render_m03, ("macro_results",)),
    ("M04", render_m04, ("macro_results",)),
    ("M05", render_m05, ("macro_results",)),
    ("M06", render_m06, ("sim_results",)),
)


def render_all_figures(
    *,
    bellman_results=None,
    policy_results=None,
    macro_results=None,
    sim_results=None,
    comparison=None,
    bank=None,
    output_dir,
    style=None,
) -> FiguresManifest:
    """Render the full figure set into ``output_dir``.

    Every figure id in :data:`FIGURE_IDS` gets a :class:`FigureRecord`: ``ok``
    with a file on disk, or ``skipped`` with a stable ``missing_reason`` and no
    file.  Multi-episode inputs share one horizontal episode axis and one
    ``b``/``z``/colour scale.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for source in (bellman_results, policy_results, macro_results, sim_results):
        all_results.extend(_as_episode_map(source).values())
    shared_style = resolve_style(style, all_results)

    annotations = {
        "Macro Anchor": _describe_macro_anchor(bank),
        "数据分布来源": _describe_bank_source(bank),
        "资本加权": "各图见自身口径",
        "灭绝 path": _describe_extinct(sim_results),
    }

    inputs = {
        "bellman_results": bellman_results,
        "policy_results": policy_results,
        "macro_results": macro_results,
        "sim_results": sim_results,
        "comparison": comparison,
    }
    manifest = FiguresManifest(output_dir=str(output_dir), style=shared_style)
    for figure_id, renderer, wanted in _SINGLE_RENDERERS:
        kwargs = {"output_dir": output_dir, "style": shared_style}
        for name in wanted:
            kwargs[name] = inputs.get(name)
        kwargs["annotations"] = annotations
        try:
            record = renderer(**kwargs)
        except Exception as exc:  # never fake a successful figure on error
            record = _skip(
                figure_id,
                f"RENDER_ERROR:{type(exc).__name__}",
                {"error": str(exc)[:400]},
            )
            record.caption = CAPTIONS.get(figure_id, "")
        record.meta.setdefault("n_samples", record.n_samples)
        for _key, _value in _style_meta(shared_style).items():
            record.meta.setdefault(_key, _value)
        manifest.figures[figure_id] = record
    return manifest


__all__ = [
    "CANONICAL_3D_NAMES",
    "CAPTIONS",
    "FIGURE_FILENAMES",
    "FIGURE_IDS",
    "FigureRecord",
    "FigureStyle",
    "FiguresManifest",
    "grid_axes",
    "render_all_figures",
    "render_f01",
    "render_f02",
    "render_f03",
    "render_f04",
    "render_f05",
    "render_f06",
    "render_f07",
    "render_f08",
    "render_f09",
    "render_f10",
    "render_f11_waterfall",
    "render_f12",
    "render_m01",
    "render_m02",
    "render_m03",
    "render_m04",
    "render_m05",
    "render_m06",
    "render_p01",
    "render_p02",
    "render_p03",
    "render_p04",
    "render_v01",
    "render_v02",
    "render_v03",
    "render_v04",
    "render_v05",
    "render_v06",
    "resolve_style",
]
