"""Obsidian-compatible convergence report builder.

``build_convergence_report`` renders ``convergence_report.md`` from the artifacts
of one ``run_convergence_evaluation`` run.  Contract:

* 14 fixed sections (Checkpoint/Episode, EvalBank provenance, Value, Policy,
  Bellman residual, FC1/SDF, Macro, Distribution, Capital Waterfall, Outer
  Fixed Point, convergence verdict, failure reasons, Gomes-mechanism
  comparison, next steps);
* every mathematical formula is a **single-line** ``$$...$$`` block so that
  Obsidian (and other single-line renderers) never see a split formula;
* every figure is referenced by its ``figures/<file>.png`` relative path and
  every table by its ``metrics/`` / ``tables/`` relative path;
* a missing figure / metric group / checkpoint is never left blank: it is
  listed with an explicit reason code (``NOT_EVALUATED`` /
  ``INSUFFICIENT_EVIDENCE`` / the concrete evaluator or render reason code).
  Nothing is fabricated.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

__all__ = ["SECTION_TITLES", "build_convergence_report"]

SECTION_TITLES: Tuple[str, ...] = (
    "Checkpoint / Episode 信息",
    "数据来源与 EvalBank",
    "Value 形态",
    "Policy 形态",
    "Bellman Residual（frozen / fresh 双口径）",
    "FC1 / SDF",
    "Macro Dynamics",
    "Distribution Dynamics",
    "Capital Waterfall",
    "Outer Fixed Point（六组指标 + Gate 状态）",
    "收敛判断",
    "当前主要失败原因",
    "与 Gomes 经济机制对照",
    "下一步建议",
)

REASON_NOT_EVALUATED = "NOT_EVALUATED"
REASON_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
REASON_THRESHOLD_NOT_CONFIGURED = "THRESHOLD_NOT_CONFIGURED"
REASON_NO_FIGURE = "NO_FIGURE_REGISTERED"

_FIGURE_BY_SECTION: Dict[int, Tuple[str, ...]] = {
    3: ("V01", "V02", "V03", "V05", "F01"),
    4: ("P01", "P02", "P03", "P04", "F04", "F02"),
    5: ("F05", "V04", "F03"),
    6: ("M01", "M04", "M05"),
    7: ("M02", "M03", "M06", "F07", "F08"),
    8: ("F09", "F10"),
    9: ("F11",),
    10: ("F12", "F06", "V06"),
    13: ("F02", "F04", "F09", "F10", "M02", "M03", "M04", "M06"),
}

_TABLE_BY_SECTION: Dict[int, Tuple[str, ...]] = {
    1: ("manifest.json",),
    2: ("bank/bank_manifest.json",),
    3: ("metrics/value_drift.csv",),
    4: ("metrics/policy_drift.csv",),
    5: ("metrics/bellman_metrics.csv",),
    6: ("metrics/macro_metrics.csv", "metrics/sdf_metrics.csv"),
    7: ("tables/macro_state_table.csv",),
    8: ("metrics/distribution_metrics.csv",),
    9: ("metrics/simulation_metrics.csv",),
    10: ("metrics/convergence_status.json", "tables/episode_comparison.csv"),
    11: ("metrics/convergence_status.json",),
    12: ("diagnostics/failure_report.json",),
    14: ("diagnostics/data_coverage.json", "tables/summary_table.csv"),
}


# ---------------------------------------------------------------------------
# small formatting helpers
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return "NaN"
        if number == int(number) and abs(number) < 1e15:
            return str(int(number))
        return f"{number:.6g}"
    text = str(value)
    return text if text else "—"


def _short(value: Any, width: int = 24) -> str:
    text = _fmt(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def _reason(code: Any) -> str:
    return f"（原因码：`{_fmt(code)}`）"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "| " + " | ".join(str(name) for name in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_fmt(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, sep] + body)


def _result_get(result: Any, name: str, default: Any = None) -> Any:
    if result is None:
        return default
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _result_metrics(result: Any) -> Dict[str, Any]:
    metrics = _result_get(result, "metrics", {}) or {}
    return dict(metrics)


def _result_tables(result: Any) -> Dict[str, pd.DataFrame]:
    tables = _result_get(result, "tables", {}) or {}
    return dict(tables)


def _frame_value(frame: Optional[pd.DataFrame], column: str, default: Any = None) -> Any:
    if frame is None or frame.empty or column not in frame.columns:
        return default
    return frame.iloc[-1].get(column, default)


def _comparison_table(comparison: Any, name: str) -> Optional[pd.DataFrame]:
    metrics = _result_get(comparison, "metrics", {}) or {}
    frame = metrics.get(name)
    if frame is None or not isinstance(frame, pd.DataFrame):
        return None
    return frame


def _status_of(frame: Optional[pd.DataFrame], episode: Any) -> Tuple[str, str]:
    if frame is None or frame.empty or "episode" not in frame.columns:
        return REASON_NOT_EVALUATED, REASON_NOT_EVALUATED
    rows = frame.loc[frame["episode"] == episode]
    if rows.empty:
        return REASON_NOT_EVALUATED, "NO_RECORD_FOR_EPISODE"
    row = rows.iloc[-1]
    return _fmt(row.get("status", REASON_NOT_EVALUATED)), _fmt(
        row.get("reason_code", REASON_NOT_EVALUATED)
    )


def _figure_manifest_map(figures_manifest: Any) -> Dict[str, Any]:
    figures = _result_get(figures_manifest, "figures", {}) or {}
    return dict(figures)


def _figure_lines(figures_manifest: Any, figure_ids: Sequence[str]) -> List[str]:
    registry = _figure_manifest_map(figures_manifest)
    lines: List[str] = []
    for figure_id in figure_ids:
        record = registry.get(figure_id)
        if record is None:
            lines.append(f"- `{figure_id}`：未注册 {_reason(REASON_NO_FIGURE)}")
            continue
        path = _result_get(record, "path")
        caption = _result_get(record, "caption", "") or ""
        status = _result_get(record, "status", "skipped")
        if status == "ok" and path:
            relative = Path(path).name
            lines.append(f"- `{figure_id}`：![{figure_id}](figures/{relative})")
            if caption:
                lines.append(f"  - 说明：{caption}")
        else:
            reason = _result_get(record, "missing_reason") or REASON_NOT_EVALUATED
            lines.append(f"- `{figure_id}`：缺失 {_reason(reason)}")
    return lines


def _table_lines(section: int) -> List[str]:
    return [f"- 表：`{name}`" for name in _TABLE_BY_SECTION.get(section, ())]


def _metric_row(
    frame: Optional[pd.DataFrame],
    keys: Sequence[str],
    episodes: Sequence[Any],
    *,
    label: str = "指标",
) -> List[str]:
    """Episode-axis matrix of metric keys (absolute errors + sample sizes)."""
    available: List[str] = []
    if frame is not None and not frame.empty:
        available = [name for name in keys if name in frame.columns]
    if not available:
        return [
            f"{label}：未评价 {_reason(REASON_NOT_EVALUATED)}"
            f"（期望列：{', '.join(keys)}）"
        ]
    rows: List[Sequence[Any]] = []
    for name in available:
        row: List[Any] = [name]
        for episode in episodes:
            if frame is None or "episode" not in frame.columns:
                row.append("—")
                continue
            subset = frame.loc[frame["episode"] == episode]
            row.append(_fmt(subset.iloc[-1][name]) if not subset.empty else "—")
        rows.append(row)
    return [_md_table([label] + [f"ep{_fmt(ep)}" for ep in episodes], rows)]


def _pick(frame: Optional[pd.DataFrame], candidates: Sequence[str]) -> List[str]:
    if frame is None or frame.empty:
        return []
    return [name for name in candidates if name in frame.columns]


# ---------------------------------------------------------------------------
# report builder
# ---------------------------------------------------------------------------


def build_convergence_report(
    *,
    manifest: Mapping[str, Any],
    comparison: Any,
    figures_manifest: Any,
    results_by_episode: Mapping[Any, Mapping[str, Any]],
    bank_manifest: Mapping[str, Any],
    output_path: str | Path,
) -> str:
    """Render ``convergence_report.md`` and return the markdown text.

    ``comparison`` is a :class:`~evaluation.convergence.comparator.ComparisonResult`,
    ``figures_manifest`` a
    :class:`~evaluation.convergence.figures.FiguresManifest` and
    ``results_by_episode`` maps episode id to ``{"bellman": ..., "policy": ...,
    "macro": ..., "simulation": ...}`` (any entry may be ``None``).
    """
    results: Dict[int, Mapping[str, Any]] = {}
    for key, value in dict(results_by_episode or {}).items():
        try:
            results[int(key)] = dict(value or {})
        except (TypeError, ValueError):
            continue
    episodes = sorted(results.keys())
    if not episodes:
        episodes = sorted(
            {
                int(entry.get("episode"))
                for entry in (manifest.get("checkpoints") or [])
                if entry.get("episode") is not None
            }
        )

    status_payload = _result_get(comparison, "convergence_status", {}) or {}
    failure_entries = (manifest.get("evaluation") or {}).get("failures") or []

    lines: List[str] = []
    lines.append("# 均衡收敛评估报告（convergence_report.md）")
    lines.append("")
    lines.append(
        f"- run_id：`{_fmt(manifest.get('run_id'))}`；输出目录：`{_fmt(manifest.get('run_dir'))}`"
    )
    lines.append(
        f"- 生成时间（UTC）：{_fmt(manifest.get('created_utc'))}；"
        f"evaluation 代码 commit：`{_fmt(manifest.get('code_commit'))}`"
    )
    lines.append(
        f"- 评价口径：Evaluation-Only（optimizer_steps="
        f"{_fmt((manifest.get('param_hash') or {}).get('optimizer_steps', 0))}，"
        f"参数哈希前后一致="
        f"{_fmt((manifest.get('param_hash') or {}).get('all_unchanged'))}）"
    )
    lines.append(
        "- 本报告只汇总实际产出的证据：缺失的图 / 指标 / checkpoint 一律列明原因码，"
        "不补画、不伪造。"
    )
    lines.append("")

    # ---- 1. Checkpoint / Episode -----------------------------------------
    lines.append(f"## 1. {SECTION_TITLES[0]}")
    lines.append("")
    lines.append(
        "每个外层 episode 对应一个 combined checkpoint；裸 `epN_policy_value.pt` 因无法自证 "
        "Q 语义被拒绝加载（原因码 `NOT_COMBINED_CHECKPOINT`）。Q 发行定价口径一律 "
        "$$Q_{\\mathrm{claim}} = b_{+}\\cdot q_{\\mathrm{unit}}$$ "
        "（禁止 effective / 回收值混入）；旧 checkpoint 缺 `q_parameterization` / "
        "`q_recovery_normalization_mode` 时按 legacy 标注而非静默默认。"
    )
    lines.append("")
    checkpoints = list(manifest.get("checkpoints") or [])
    if checkpoints:
        rows = []
        for entry in checkpoints:
            legacy_q = "是" if entry.get("q_parameterization_assumed_legacy") else "否"
            legacy_r = "是" if entry.get("q_recovery_normalization_mode_assumed_legacy") else "否"
            rows.append(
                [
                    entry.get("episode"),
                    _short(entry.get("checkpoint_path"), 42),
                    _short(entry.get("checkpoint_sha256"), 12),
                    _short(entry.get("git_commit"), 12),
                    entry.get("q_parameterization"),
                    legacy_q,
                    entry.get("q_recovery_normalization_mode"),
                    legacy_r,
                    (entry.get("value_parameterization") or {}).get("mode"),
                    entry.get("status", "OK"),
                    entry.get("reason_code", "EVALUATED"),
                ]
            )
        lines.append(
            _md_table(
                [
                    "episode",
                    "checkpoint",
                    "sha256",
                    "git_commit",
                    "q_parameterization",
                    "Q legacy 标注",
                    "q_recovery_normalization_mode",
                    "recovery legacy 标注",
                    "value_parameterization",
                    "status",
                    "reason_code",
                ],
                rows,
            )
        )
    else:
        lines.append(f"无 checkpoint 记录 {_reason(REASON_NOT_EVALUATED)}")
    lines.append("")
    first_value = (checkpoints[0].get("value_parameterization") or {}) if checkpoints else {}
    lines.append(
        f"- value_parameterization：`{_fmt(first_value.get('scale_formula'))}`；"
        "physical 与 normalized 双口径同报，normalized = R / equity_value_scale。"
    )
    lines.extend(_table_lines(1))
    lines.append("")

    # ---- 2. EvalBank -----------------------------------------------------
    lines.append(f"## 2. {SECTION_TITLES[1]}")
    lines.append("")
    source = dict(bank_manifest.get("source") or {})
    source_kind = source.get("kind")
    lines.append(
        f"- bank_version：`{_fmt(bank_manifest.get('bank_version'))}`；"
        f"config_snapshot_hash：`{_short(bank_manifest.get('config_snapshot_hash'), 16)}`；"
        f"构建 commit：`{_short(bank_manifest.get('evaluation_code_commit'), 12)}`"
    )
    if source_kind == "reference-policy-distribution":
        lines.append(
            "- **reference-policy-distribution 标记**：Bank A 的自然分布来自 baseline "
            "参考策略模拟（非各集 on-policy 分布），跨 episode 逐字节一致；"
            "不得解读为当前集自身的稳态分布。"
        )
    else:
        lines.append(
            f"- 数据来源 kind：`{_fmt(source_kind)}`"
            f"{'' if source_kind else _reason(REASON_NOT_EVALUATED)}"
        )
    lines.append(
        f"- checkpoint_ref：`{_short(source.get('checkpoint_ref'), 24)}`；"
        f"train_lineage_exclusion：`{_fmt(source.get('train_lineage_exclusion'))}`"
    )
    if bank_manifest.get("bank_version") not in (None, "evaluation_bank_v1"):
        lines.append(
            f"- ⚠️ bank_version 非 `evaluation_bank_v1`（实际 "
            f"`{_fmt(bank_manifest.get('bank_version'))}`）：与 v1 结果不可比"
            "（原因码 `INCOMPARABLE_BANK_VERSION`）。"
        )
    lines.append("")
    bank_rows = []
    for bank_key in sorted((bank_manifest.get("banks") or {}).keys()):
        info = (bank_manifest.get("banks") or {}).get(bank_key) or {}
        eta_share = dict(info.get("eta_share") or {})
        source_share = dict(info.get("source_share") or {})
        bank_rows.append(
            [
                bank_key,
                _short(info.get("file"), 28),
                _short(info.get("sha256"), 12),
                info.get("n_rows"),
                _fmt(eta_share.get("eta1")),
                _fmt(source_share.get("simulated")),
                _fmt(source_share.get("coverage")),
                info.get("seed"),
            ]
        )
    if bank_rows:
        lines.append(
            _md_table(
                ["bank", "file", "sha256", "n_rows", "eta=1 占比", "source=simulated", "source=coverage", "seed"],
                bank_rows,
            )
        )
    else:
        lines.append(f"EvalBank 清单为空 {_reason(REASON_NOT_EVALUATED)}")
    lines.append("")
    lines.append(
        "- Bank 语义：A=固定自然分布（η 不重平衡）、B=A 的 η∈{0,1} 平衡反事实视图、"
        "C=固定边界/应力网格（不依赖任何单集网络预测）、D=独立宏观 holdout（按键 join）、"
        "E=固定 rollout（`stable_firm_identity` keyed RNG 冲击 + 三种初始杠杆）。"
    )
    lines.extend(_table_lines(2))
    lines.append("")

    def _results_of(episode: int, block: str) -> Any:
        return (results.get(episode) or {}).get(block)

    latest = episodes[-1] if episodes else None

    # ---- 3. Value --------------------------------------------------------
    lines.append(f"## 3. {SECTION_TITLES[2]}")
    lines.append("")
    lines.append(
        "口径：physical = equity_value_scale × normalized，exp_xz 模式下 "
        "$$s(x,z) = 1 + \\exp(\\mathrm{clamp}(x+z,\\max=\\log\\_\\max))$$ "
        "；残差 normalized = R / s(x,z)。Value 曲面 P0/PI/P 与发行求偿权 Q_claim 来自固定 "
        "Bank C 网格（b×z），η=0/η=1 两个反事实视图并列。"
    )
    lines.append("")
    value_frame = _comparison_table(comparison, "value_drift")
    if latest is not None and value_frame is not None and not value_frame.empty:
        quantities = [
            "p0_physical",
            "pi_physical",
            "p0_normalized",
            "pi_normalized",
            "P",
            "Q_claim",
        ]
        rows = []
        subset = value_frame.loc[value_frame["episode"] == latest]
        for quantity in quantities:
            columns = [
                f"{quantity}_mae",
                f"{quantity}_relative_mae",
                f"{quantity}_rmse",
                f"{quantity}_p99",
                f"{quantity}_max",
                f"{quantity}_n",
            ]
            if not any(name in value_frame.columns for name in columns):
                rows.append([quantity, REASON_NOT_EVALUATED, "—", "—", "—", "—", "—"])
                continue
            row = [quantity]
            for name in columns:
                row.append(
                    _fmt(subset.iloc[-1][name])
                    if name in value_frame.columns and not subset.empty
                    else "—"
                )
            rows.append(row)
        lines.append(f"Value 漂移（相邻 episode，固定评价状态集；最新 ep{_fmt(latest)}）：")
        lines.append("")
        lines.append(
            _md_table(
                ["quantity", "mae(绝对误差)", "relative_mae", "rmse", "p99", "max", "n(样本量)"],
                rows,
            )
        )
    else:
        lines.append(
            f"Value 漂移表缺失 {_reason(REASON_INSUFFICIENT_EVIDENCE if len(episodes) < 2 else REASON_NOT_EVALUATED)}"
            "（需要至少两个语义可比 episode 的 Bellman 结果逐行对齐）。"
        )
    lines.append("")
    lines.extend(_figure_lines(figures_manifest, _FIGURE_BY_SECTION[3]))
    lines.append("")
    lines.extend(_table_lines(3))
    lines.append("")

    # ---- 4. Policy -------------------------------------------------------
    lines.append(f"## 4. {SECTION_TITLES[3]}")
    lines.append("")
    lines.append(
        "正式策略 = Grid BP（`BPGridTeacher` 在候选杠杆网格上的贪心解）；bp0/bpI/bp 头输出仅兼容列。"
        "融资门：$$b_{t+1} = \\eta_t\\, bp^{\\star} + (1-\\eta_t)\\, b_t$$ "
        "（η=0 强制 b_next=b，η=1 时 b_next=bp*）。bar_z = 1-sigmoid(Phat/τ_z) 是软违约概率，"
        "不是生产率门槛；硬违约边界由 Phat(b,z,η,m)=0 求出。Bar_i ≡ Bar_i_executed ≠ Bar_i_model"
        "（entry 节点 executed 强制 0）。"
    )
    lines.append("")
    policy_frame = _comparison_table(comparison, "policy_drift")
    policy_keys = [
        "bp_star_drift_max",
        "policy_flip_rate",
        "default_flip_rate",
        "investment_flip_rate",
        "n_eta1_rows",
    ]
    if policy_frame is not None and not policy_frame.empty:
        present = _pick(policy_frame, policy_keys)
        if present:
            lines.extend(_metric_row(policy_frame, present, episodes, label="策略漂移指标"))
        else:
            lines.append(f"策略漂移指标列缺失 {_reason(REASON_NOT_EVALUATED)}")
        status, reason = (
            _status_of(policy_frame, latest)
            if latest is not None
            else (REASON_NOT_EVALUATED, REASON_NOT_EVALUATED)
        )
        lines.append(
            f"- 最新 episode（ep{_fmt(latest)}）policy_drift 状态：`{status}` {_reason(reason)}"
        )
    else:
        lines.append(
            f"policy_drift 表缺失 {_reason(REASON_INSUFFICIENT_EVIDENCE if len(episodes) < 2 else REASON_NOT_EVALUATED)}"
        )
    lines.append("")
    for episode in episodes:
        policy = _results_of(episode, "policy")
        if policy is None:
            lines.append(
                f"- ep{_fmt(episode)}：Policy 评价缺失 {_reason(REASON_NOT_EVALUATED)}"
                "（未运行或失败，见第 12 节）"
            )
            continue
        metrics = _result_metrics(policy)
        lines.append(
            f"- ep{_fmt(episode)}：policy_rows_n=`{_fmt(metrics.get('policy_rows_n'))}`、"
            f"eta1_financing_gate_max_error_mix=`{_fmt(metrics.get('eta1_financing_gate_max_error_mix'))}`；"
            "投资边界唯一性分类与硬/软违约边界对照见 F02/F04 与 `metrics/policy_drift.csv`。"
        )
    lines.append("")
    lines.extend(_figure_lines(figures_manifest, _FIGURE_BY_SECTION[4]))
    lines.append("")
    lines.extend(_table_lines(4))
    lines.append("")

    # ---- 5. Bellman residual --------------------------------------------
    lines.append(f"## 5. {SECTION_TITLES[4]}")
    lines.append("")
    lines.append(
        "两个残差严格分开：frozen 残差 "
        "$$R^{\\mathrm{frozen}}_i = P_i - \\mathrm{CF}^{\\mathrm{frozen}}_i - \\sum_j w_j M_j P'_j$$ "
        "是训练冻结策略下的 Bellman 恒等式残差；fresh-greedy 残差 "
        "$$R^{\\mathrm{fresh}} = P - V^{\\mathrm{greedy}}(\\mathrm{Grid\\text{-}BP~re\\text{-}solve})$$ "
        "**fresh 才是最优性证据**（训练 loss 不作收敛证据）。发行/残差一律 claim 口径；"
        "Physical 与 Normalized 同报。Q 区域残差按 b=0 区 / 违约回收区 / 存活区 / 边界区"
        "（Q0/QD/QS/Polish）分层。"
    )
    lines.append("")
    bellman_frame = _comparison_table(comparison, "firm_bellman")
    bellman_keys = [
        "fresh_p0_normalized_mae",
        "fresh_pi_normalized_mae",
        "fresh_p0_physical_mae",
        "fresh_pi_physical_mae",
        "frozen_p0_normalized_mae",
        "frozen_pi_normalized_mae",
        "frozen_p0_physical_mae",
        "frozen_pi_physical_mae",
        "q_qs_claim_bellman_trainM_mae",
        "q_qd_claim_bellman_trainM_mae",
        "q_q0_claim_bellman_trainM_mae",
        "q_polish_claim_bellman_trainM_mae",
        "parity_max_abs_error",
        "frozen_p0_physical_simulated_n",
    ]
    if bellman_frame is not None and not bellman_frame.empty:
        present = _pick(bellman_frame, bellman_keys)
        if present:
            lines.extend(_metric_row(bellman_frame, present, episodes, label="Bellman 指标"))
        else:
            lines.append(f"Bellman 指标列缺失 {_reason(REASON_NOT_EVALUATED)}")
    else:
        lines.append(f"firm_bellman 表缺失 {_reason(REASON_NOT_EVALUATED)}")
    lines.append("")
    for episode in episodes:
        bellman = _results_of(episode, "bellman")
        if bellman is None:
            lines.append(
                f"- ep{_fmt(episode)}：Bellman 评价缺失 {_reason(REASON_NOT_EVALUATED)}"
            )
            continue
        metrics = _result_metrics(bellman)
        lines.append(
            f"- ep{_fmt(episode)}：fresh_p0_normalized_mae=`{_fmt(metrics.get('fresh_p0_normalized_mae'))}`、"
            f"frozen_p0_normalized_mae=`{_fmt(metrics.get('frozen_p0_normalized_mae'))}`、"
            f"optimizer_steps=`{_fmt(metrics.get('optimizer_steps'))}`"
        )
    lines.append("")
    lines.extend(_figure_lines(figures_manifest, _FIGURE_BY_SECTION[5]))
    lines.append("")
    lines.extend(_table_lines(5))
    lines.append("")

    # ---- 6. FC1 / SDF ----------------------------------------------------
    lines.append(f"## 6. {SECTION_TITLES[5]}")
    lines.append("")
    lines.append(
        "FC1 两口径分开报：production one-step（每次喂父节点算得宏观态，正式模拟调用规则）与 "
        "autonomous recursive（完全自回归）"
        "$$\\hat{x}_{t+h} = f^{h}(\\hat{x}_{t}),\\quad h \\in \\{1,2,5,10,20,50\\}$$ "
        "；hatcf/lnkf（belief）≠ Hatc/LnK（实现聚合），同行对齐即 shift=0。SDF 报 M 均值/标准误、"
        "条件矩约束违反量与 Signed AiO 估计（负样本均值 ≠ 负总体条件矩，解读有护栏）。"
    )
    lines.append("")
    macro_frame = _comparison_table(comparison, "macro")
    fc1_keys = [
        "fc1_one_step_hatc_mae",
        "fc1_one_step_lnk_mae",
        "fc1_one_step_hatc_bias",
        "fc1_model_one_step_hatc_mae",
        "fc1_rec_production_one_step_hatc_h5_bias",
        "fc1_rec_autonomous_recursive_hatc_h5_bias",
        "fc1_rec_autonomous_recursive_hatc_h50_rmse",
    ]
    sdf_keys = [
        "sdf_g_max",
        "sdf_population_target_abs",
        "sdf_nonfinite_ratio",
        "sdf_signed_aio_estimate",
        "sdf_signed_aio_se",
        "sdf_M_mean",
        "sdf_M_mean_se",
        "sdf_n_parents",
    ]
    if macro_frame is not None and not macro_frame.empty:
        present_fc1 = _pick(macro_frame, fc1_keys)
        present_sdf = _pick(macro_frame, sdf_keys)
        if present_fc1:
            lines.append("FC1 指标（production one-step 与 autonomous recursive 分列于指标名中）：")
            lines.append("")
            lines.extend(_metric_row(macro_frame, present_fc1, episodes, label="FC1 指标"))
        else:
            lines.append(f"FC1 指标缺失 {_reason(REASON_NOT_EVALUATED)}")
        lines.append("")
        if present_sdf:
            lines.extend(_metric_row(macro_frame, present_sdf, episodes, label="SDF 指标"))
        else:
            lines.append(f"SDF 指标缺失 {_reason(REASON_NOT_EVALUATED)}")
    else:
        lines.append(f"macro/sdf 指标表缺失 {_reason(REASON_NOT_EVALUATED)}")
    lines.append("")
    lines.extend(_figure_lines(figures_manifest, _FIGURE_BY_SECTION[6]))
    lines.append("")
    lines.extend(_table_lines(6))
    lines.append("")

    # ---- 7. Macro dynamics ----------------------------------------------
    lines.append(f"## 7. {SECTION_TITLES[6]}")
    lines.append("")
    lines.append(
        "宏观动态按 Recession/Normal/Expansion（x 分位切分）与资本分层报告；递归误差的斜率即"
        "信念漂移速度，出训练范围比例决定长期模拟可信度。跨 episode 的宏观轨迹漂移为 "
        "`macro_trajectory_drift_abs_max`。"
    )
    lines.append("")
    macro_state_rows: List[Sequence[Any]] = []
    for episode in episodes:
        macro = _results_of(episode, "macro")
        metrics = _result_metrics(macro)
        for state in ("recession", "normal", "expansion"):
            key_mae = f"fc1_one_step_hatc_mae_{state}"
            key_n = f"fc1_one_step_hatc_n_{state}"
            if macro is None or key_mae not in metrics:
                macro_state_rows.append([episode, state, REASON_NOT_EVALUATED, "—", "—", "—"])
                continue
            macro_state_rows.append(
                [
                    episode,
                    state,
                    "OK",
                    _fmt(metrics.get(key_mae)),
                    _fmt(metrics.get(f"fc1_one_step_lnk_mae_{state}")),
                    _fmt(metrics.get(key_n)),
                ]
            )
    if macro_state_rows:
        lines.append(
            _md_table(
                ["episode", "macro_state", "status", "hatc_mae", "lnk_mae", "n"],
                macro_state_rows,
            )
        )
    else:
        lines.append(f"宏观分态表缺失 {_reason(REASON_NOT_EVALUATED)}")
    lines.append("")
    if macro_frame is not None and "macro_trajectory_drift_abs_max" in getattr(
        macro_frame, "columns", []
    ):
        drift_row = [
            [
                episode,
                _fmt(
                    _frame_value(
                        macro_frame.loc[macro_frame["episode"] == episode],
                        "macro_trajectory_drift_abs_max",
                    )
                ),
            ]
            for episode in episodes
        ]
        lines.append(_md_table(["episode", "macro_trajectory_drift_abs_max"], drift_row))
    else:
        lines.append(
            f"macro_trajectory_drift_abs_max 缺失 {_reason(REASON_INSUFFICIENT_EVIDENCE)}"
            "（需要相邻可比 episode 的 macro 结果）。"
        )
    lines.append("")
    lines.extend(_figure_lines(figures_manifest, _FIGURE_BY_SECTION[7]))
    lines.append("")
    lines.extend(_table_lines(7))
    lines.append("")

    # ---- 8. Distribution dynamics ---------------------------------------
    lines.append(f"## 8. {SECTION_TITLES[7]}")
    lines.append("")
    lines.append(
        "**选择效应声明**：simulate 面板只写存活行（退出 = 行消失，判据 bar_z≥0.5），"
        "灭绝 path 必须保留在统计中（`path_status` 逐期记录 active/extinct），"
        "本报告的分布统计不做幸存者偏差修正，只显式标注灭绝条目。分布距离用 "
        "Wasserstein-1（b/z 边缘）与固定网格二维直方图 L1 距离（不做 2D 最优传输）。"
    )
    lines.append("")
    distribution_frame = _comparison_table(comparison, "distribution")
    distribution_keys = [
        "distribution_drift_abs_max",
        "dist_n_slices",
        "dist_min_active_paths",
        "dist_max_active_paths",
        "dist_extinct_paths_total",
    ]
    if distribution_frame is not None and not distribution_frame.empty:
        present = _pick(distribution_frame, distribution_keys)
        if present:
            lines.extend(_metric_row(distribution_frame, present, episodes, label="分布指标"))
        else:
            lines.append(f"分布指标列缺失 {_reason(REASON_NOT_EVALUATED)}")
    else:
        lines.append(f"distribution 表缺失 {_reason(REASON_NOT_EVALUATED)}")
    lines.append("")
    for episode in episodes:
        simulation = _results_of(episode, "simulation")
        if simulation is None:
            lines.append(
                f"- ep{_fmt(episode)}：模拟分布缺失 {_reason(REASON_NOT_EVALUATED)}"
            )
            continue
        metrics = _result_metrics(simulation)
        lines.append(
            f"- ep{_fmt(episode)}：sim_n_extinct_paths_max=`{_fmt(metrics.get('sim_n_extinct_paths_max'))}`、"
            f"sim_baseline_dist_min_active_paths=`{_fmt(metrics.get('sim_baseline_dist_min_active_paths'))}`；"
            "灭绝 path id 见 sim 结果 `path_status` 表与 F09/F10 图注。"
        )
    lines.append("")
    lines.extend(_figure_lines(figures_manifest, _FIGURE_BY_SECTION[8]))
    lines.append("")
    lines.extend(_table_lines(8))
    lines.append("")

    # ---- 9. Capital waterfall -------------------------------------------
    lines.append(f"## 9. {SECTION_TITLES[8]}")
    lines.append("")
    lines.append(
        "资本账本恒等式（`capital_growth_decomposition` 口径）："
        "$$K_{t+1} = K_t + \\Delta K^{\\mathrm{incumbent}} + K^{\\mathrm{entry\\_endpoint}} - K^{\\mathrm{exit}} + \\varepsilon^{\\mathrm{acct}}$$ "
        "同节点进出企业是备忘行（same_node_entrant_exit），不进恒等式。"
    )
    lines.append("")
    waterfall_rows: List[Sequence[Any]] = []
    for episode in episodes:
        simulation = _results_of(episode, "simulation")
        tables = _result_tables(simulation)
        waterfall = None
        for name, frame in tables.items():
            if (
                name.startswith("capital_waterfall")
                and not name.endswith("totals")
                and isinstance(frame, pd.DataFrame)
                and not frame.empty
            ):
                waterfall = frame
                break
        if waterfall is None:
            waterfall_rows.append([episode, REASON_NOT_EVALUATED, "—", "—", "—", "—"])
            continue
        totals = waterfall.select_dtypes(include="number").sum(numeric_only=True)
        residual_column = "accounting_residual"
        residual = (
            float(waterfall[residual_column].abs().max())
            if residual_column in waterfall.columns
            else float("nan")
        )
        waterfall_rows.append(
            [
                episode,
                "OK",
                _fmt(totals.get("initial_capital")),
                _fmt(totals.get("incumbent_dk")),
                _fmt(totals.get("entrant_capital")),
                _fmt(residual),
            ]
        )
    if waterfall_rows:
        lines.append(
            _md_table(
                [
                    "episode",
                    "status",
                    "Σinitial_capital",
                    "Σincumbent_dk",
                    "Σentrant_capital",
                    "max|accounting_residual|",
                ],
                waterfall_rows,
            )
        )
    else:
        lines.append(f"Waterfall 表缺失 {_reason(REASON_NOT_EVALUATED)}")
    lines.append("")
    lines.extend(_figure_lines(figures_manifest, _FIGURE_BY_SECTION[9]))
    lines.append("")
    lines.extend(_table_lines(9))
    lines.append("")

    # ---- 10. Outer fixed point ------------------------------------------
    lines.append(f"## 10. {SECTION_TITLES[9]}")
    lines.append("")
    lines.append(
        "六组指标 Dashboard（统一 episode 轴，缺失行显式带 status/reason_code）："
        "firm_bellman、value_drift、policy_drift、macro、simulation、distribution；"
        "外加 gate-only 的 comparability 预检表。Gate 三态：PASS / FAIL / "
        "INSUFFICIENT_EVIDENCE / THRESHOLD_NOT_CONFIGURED。"
    )
    lines.append("")
    group_frames = {
        "firm_bellman": _comparison_table(comparison, "firm_bellman"),
        "value_drift": _comparison_table(comparison, "value_drift"),
        "policy_drift": _comparison_table(comparison, "policy_drift"),
        "macro": _comparison_table(comparison, "macro"),
        "simulation": _comparison_table(comparison, "simulation"),
        "distribution": _comparison_table(comparison, "distribution"),
    }
    group_rows: List[Sequence[Any]] = []
    for group, frame in group_frames.items():
        status, reason = (
            _status_of(frame, latest)
            if latest is not None
            else (REASON_NOT_EVALUATED, REASON_NOT_EVALUATED)
        )
        n_columns = int(len(frame.columns)) if frame is not None else 0
        group_rows.append([group, status, reason, n_columns])
    lines.append(
        _md_table(["指标组", "最新 episode 状态", "reason_code", "指标列数"], group_rows)
    )
    lines.append("")
    episode_status = list(status_payload.get("episode_status") or [])
    gate_rows: List[Sequence[Any]] = []
    for entry in episode_status:
        episode = entry.get("episode")
        for group, gates in (entry.get("gates") or {}).items():
            for gate_id, gate in (gates or {}).items():
                gate_rows.append(
                    [
                        episode,
                        group,
                        gate_id,
                        gate.get("status"),
                        gate.get("reason_code"),
                        _fmt(gate.get("threshold")),
                        _fmt(gate.get("value")),
                    ]
                )
    if gate_rows:
        lines.append(
            _md_table(
                ["episode", "group", "gate_id", "status", "reason_code", "threshold", "value"],
                gate_rows,
            )
        )
    else:
        lines.append(
            f"Gate 状态缺失 {_reason(REASON_NOT_EVALUATED)}"
            "（comparator 未产出 episode_status）"
        )
    lines.append("")
    lines.extend(_figure_lines(figures_manifest, _FIGURE_BY_SECTION[10]))
    lines.append("")
    lines.extend(_table_lines(10))
    lines.append("")

    # ---- 11. convergence verdict ----------------------------------------
    lines.append(f"## 11. {SECTION_TITLES[10]}")
    lines.append("")
    verdict = status_payload.get("convergence_status") or REASON_NOT_EVALUATED
    lines.append(
        f"- **收敛判断：`{_fmt(verdict)}`** {_reason(status_payload.get('reason_code'))}"
    )
    lines.append(
        f"- converged_candidate：`{_fmt(status_payload.get('converged_candidate'))}`；"
        f"最大连续满足计数 `{_fmt(status_payload.get('max_consecutive_satisfied_count'))}` / "
        f"要求 `{_fmt(status_payload.get('required_consecutive_comparable_episodes'))}`"
    )
    lines.append(
        f"- 参与 episode：`{_fmt(status_payload.get('participating_episodes'))}`；"
        f"可比 episode 对：`{_fmt(status_payload.get('comparable_episode_pairs'))}`"
    )
    lines.append(
        f"- 阈值配置完整：`{_fmt(status_payload.get('threshold_config_complete'))}`；"
        f"缺失的必要阈值：`{_fmt(status_payload.get('missing_required_thresholds'))}`"
        f"{_reason(REASON_THRESHOLD_NOT_CONFIGURED) if status_payload.get('missing_required_thresholds') else ''}"
    )
    lines.append("")
    lines.append(
        "判据说明（Converged Candidate）：连续 3 个**语义可比、固定 bank 逐字节一致**的相邻 "
        "episode 上，所有必要 gate 均 PASS，才判 `CONVERGED_CANDIDATE`；任一必要 gate FAIL 即 "
        "`NOT_CONVERGED`（`REQUIRED_GATE_FAILED`）；证据不足为 `INSUFFICIENT_EVIDENCE`。"
        "阈值为 null 的 gate 永远判 `THRESHOLD_NOT_CONFIGURED`，绝不视为 PASS（不编造阈值）。"
        "最优性证据以 fresh-greedy 残差为准，训练 loss 不作收敛证据。"
    )
    lines.append("")
    status_counts: Dict[str, int] = {}
    for entry in episode_status:
        for gates in (entry.get("gates") or {}).values():
            for gate in (gates or {}).values():
                status_name = str(gate.get("status"))
                status_counts[status_name] = status_counts.get(status_name, 0) + 1
    if status_counts:
        lines.append(
            _md_table(
                ["gate 状态", "计数"],
                sorted(status_counts.items(), key=lambda item: item[0]),
            )
        )
    else:
        lines.append(f"Gate 状态计数缺失 {_reason(REASON_NOT_EVALUATED)}")
    lines.append("")
    lines.extend(_table_lines(11))
    lines.append("")

    # ---- 12. failure reasons --------------------------------------------
    lines.append(f"## 12. {SECTION_TITLES[11]}")
    lines.append("")
    failures: List[str] = []
    for entry in episode_status:
        for group, gates in (entry.get("gates") or {}).items():
            for gate_id, gate in (gates or {}).items():
                if gate.get("status") == "FAIL":
                    failures.append(
                        f"- Gate `{gate_id}`（ep{_fmt(entry.get('episode'))}，{group}）：FAIL "
                        f"{_reason(gate.get('reason_code'))}，指标 `{_fmt(gate.get('metric'))}`="
                        f"`{_fmt(gate.get('value'))}`，阈值 `{_fmt(gate.get('threshold'))}`"
                    )
    not_configured = sorted(
        {
            gate_id
            for entry in episode_status
            for gates in (entry.get("gates") or {}).values()
            for gate_id, gate in (gates or {}).items()
            if gate.get("status") == REASON_THRESHOLD_NOT_CONFIGURED
        }
    )
    insufficient = sorted(
        {
            gate_id
            for entry in episode_status
            for gates in (entry.get("gates") or {}).values()
            for gate_id, gate in (gates or {}).items()
            if gate.get("status") == REASON_INSUFFICIENT_EVIDENCE
        }
    )
    for entry in failure_entries:
        failures.append(
            f"- Evaluator `{_fmt(entry.get('evaluator'))}`（ep{_fmt(entry.get('episode'))}）失败/跳过 "
            f"{_reason(entry.get('reason_code'))}：{_short(entry.get('detail'), 120)}"
        )
    registry = _figure_manifest_map(figures_manifest)
    skipped = [
        (figure_id, _result_get(record, "missing_reason") or REASON_NOT_EVALUATED)
        for figure_id, record in registry.items()
        if _result_get(record, "status") != "ok"
    ]
    if skipped:
        failures.append(
            f"- 缺失图 {len(skipped)} 张："
            + "；".join(f"`{figure_id}` {_reason(reason)}" for figure_id, reason in skipped[:12])
            + ("…" if len(skipped) > 12 else "")
        )
    if not_configured:
        failures.append(
            "- 未配置阈值的 gate（`THRESHOLD_NOT_CONFIGURED`，永不 PASS）："
            + ", ".join(f"`{name}`" for name in not_configured)
        )
    if insufficient:
        failures.append(
            "- 证据不足的 gate（`INSUFFICIENT_EVIDENCE`）："
            + ", ".join(f"`{name}`" for name in insufficient)
        )
    for frame_name, frame in group_frames.items():
        if frame is None or frame.empty or "status" not in getattr(frame, "columns", []):
            continue
        bad = frame.loc[frame["status"] != "OK"]
        for _, row in bad.iterrows():
            failures.append(
                f"- 指标组 `{frame_name}` ep{_fmt(row.get('episode'))}：`{_fmt(row.get('status'))}` "
                f"{_reason(row.get('reason_code'))}"
            )
    if failures:
        lines.extend(failures)
    else:
        lines.append(
            "- 无 FAIL / 无 evaluator 失败 / 无缺图；"
            f"若判据仍非 CONVERGED_CANDIDATE，则是 episode 数不足 "
            f"{_reason(REASON_INSUFFICIENT_EVIDENCE)}"
        )
    lines.append("")
    lines.extend(_table_lines(12))
    lines.append("")

    # ---- 13. Gomes comparison -------------------------------------------
    lines.append(f"## 13. {SECTION_TITLES[12]}")
    lines.append("")
    lines.append(
        "对照对象是 Gomes 的经济机制（投资/违约政策、杠杆分布、宏观冲击响应），"
        "**按当前模型的真实变量口径**说明，不做论文数字的复刻。"
    )
    lines.append("")
    lines.append("### 13.1 投资 / 违约政策（Gomes Figure 1 机制对照）")
    lines.append("")
    lines.append(
        "- Gomes Figure 1 的投资率 i*(b,z) 与违约边界 z_default(b) 在本模型中的对应量："
        "投资边界 = PI-V0 沿投资成本轴的零点（多零点/无零点显式分类），"
        "违约边界 = Phat(b,z,η,m)=0 的 z 解；软违约概率 bar_z = 1-sigmoid(Phat/τ_z) 与其分离。"
    )
    lines.append(
        "- 机制差异（真实口径）：(i) 本模型的正式债务策略是 Grid-BP 贪心解 bp*，"
        "受融资门 $$b_{t+1} = \\eta_t\\, bp^{\\star} + (1-\\eta_t)\\, b_t$$ 约束；"
        "(ii) 违约有软惩罚（τ、κ_b）与硬门双口径；"
        "(iii) 意愿投资 bar_i_model 与执行投资 bar_i_executed 分列，entry 节点 executed 强制 0。"
    )
    lines.append("")
    lines.extend(_figure_lines(figures_manifest, ("F02", "F04")))
    lines.append("")
    lines.append("### 13.2 杠杆分布（Gomes Figure 2 机制对照）")
    lines.append("")
    lines.append(
        "- 对应 F09/F10：按 x 分位的 Recession/Normal/Expansion 杠杆分布（企业计数加权 vs 资本加权）"
        "与 b-z 联合分布随时间演化。机制差异：本模型分布来自固定 rollout（Bank E 固定初始横截面与"
        "固定 keyed 冲击），非论文的 ergodic 分布；退出 = 行消失（bar_z≥0.5），选择效应显式标注"
        "（见第 8 节声明），灭绝 path 保留。"
    )
    lines.append("")
    lines.extend(_figure_lines(figures_manifest, ("F09", "F10")))
    lines.append("")
    lines.append("### 13.3 宏观冲击响应")
    lines.append("")
    lines.append(
        "- 对应 M02/M03/M04/M06 与固定冲击模拟：信念 hatcf/lnkf vs 实现 Hatc/LnK 路径、"
        "production one-step 与 autonomous recursive 两口径的递归误差、宏观聚合量 C/K、K、C、Y、I 轨迹。"
        "机制差异：本模型的宏观冲击经 FC1 信念-实现链传递（belief 状态第 5/6 列），"
        "与论文直接进入生产的 TFP 意识冲击不同；脉冲响应不可逐点对齐，只比较方向与持续性。"
    )
    lines.append("")
    lines.extend(_figure_lines(figures_manifest, ("M02", "M03", "M04", "M06")))
    lines.append("")
    lines.append("### 13.4 不复刻声明")
    lines.append("")
    lines.append(
        "- 本节只做机制映射：不复制论文的图样、参数校准或数值；"
        f"所有差异按上列真实变量口径逐条声明，未评价的对照项一律标 {_reason(REASON_NOT_EVALUATED)}。"
    )
    lines.append("")

    # ---- 14. next steps --------------------------------------------------
    lines.append(f"## 14. {SECTION_TITLES[13]}")
    lines.append("")
    suggestions: List[str] = []
    if not_configured:
        suggestions.append(
            "- 阈值未配置：用 `comparator.build_threshold_calibration` 基于多 seed 证据生成校准报告，"
            "人工确认后写入 ThresholdConfig（严禁自动生成阈值）。"
        )
    if str(verdict) in (REASON_INSUFFICIENT_EVIDENCE, "INSUFFICIENT_EVIDENCE"):
        suggestions.append(
            f"- 收敛证据不足：补足连续可比 episode（≥3，语义一致、bank 逐字节一致）"
            f"{_reason(REASON_INSUFFICIENT_EVIDENCE)}"
        )
    if any(entry.get("status") not in (None, "OK", "EVALUATED") for entry in failure_entries):
        suggestions.append(
            "- 修复失败的 evaluator（见第 12 节 reason code），随后以同一 mode 重跑。"
        )
    if skipped:
        suggestions.append("- 补齐缺图所需输入（对应 evaluator 结果），重跑 figures 渲染。")
    for entry in episode_status:
        for gates in (entry.get("gates") or {}).values():
            for gate_id, gate in (gates or {}).items():
                if gate.get("status") == "FAIL" and gate_id.startswith("firm_"):
                    suggestions.append(
                        f"- `{gate_id}` 超阈：优先看 fresh-greedy 残差与 Grid-BP 漂移（F05/F06），"
                        "确认是收敛未到还是评价口径问题。"
                    )
                if gate.get("status") == "FAIL" and gate_id.startswith("macro_"):
                    suggestions.append(
                        f"- `{gate_id}` 超阈：检查 FC1 递归漂移与 SDF 矩约束（M04/M05），"
                        "必要时缩短模拟 horizon 或复核出训练范围比例。"
                    )
    suggestions.append(
        "- 保持 Evaluation-Only：任何补跑都不得触碰 checkpoint（参数哈希前后一致、optimizer_steps=0）。"
    )
    suggestions.append(
        "- 若需改 bank（数据源/种子/行数），升版 `evaluation_bank_v2` 并在报告声明跨版本不可比。"
    )
    seen: set = set()
    unique_suggestions: List[str] = []
    for suggestion in suggestions:
        if suggestion not in seen:
            seen.add(suggestion)
            unique_suggestions.append(suggestion)
    lines.extend(unique_suggestions)
    lines.append("")
    lines.extend(_table_lines(14))
    lines.append("")

    markdown = "\n".join(lines).rstrip() + "\n"
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
    return markdown
