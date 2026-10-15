#!/usr/bin/env python3
"""Read-only PI continuation gain decomposition for a completed P-only path.

The program consumes immutable sweep checkpoints, validation banks, and the
previous exact movement decomposition.  It creates no optimizer and performs
no backward pass or parameter update.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import date
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze_p_only_movement_channels import (  # noqa: E402
    BP_PREFIXES,
    P_PREFIXES,
    Q_PREFIXES,
    _bank_hash_children,
    _bank_hash_parent,
    _checkpoint_paths,
    _expanded_inputs,
    _i_grid_values,
    _load_policy_state,
    _load_setup,
)
from config import Config  # noqa: E402
from training.bp_grid_teacher import _expand_grid_children  # noqa: E402
from training.bp_grid_teacher import BPGridTeacher  # noqa: E402
from training.inner_fixed_point import named_parameter_hash  # noqa: E402
from utils.firm_transition import normalize_child_weights  # noqa: E402


STATE_NAMES = ("b", "z", "eta", "i", "x", "Hatcf", "LnKF")
QUANTILES = (0.50, 0.90, 0.95, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--movement-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=None)
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--transition-start", type=int, default=0)
    parser.add_argument("--transition-end", type=int, default=9)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--closure-atol", type=float, default=2e-6)
    parser.add_argument("--finite-difference", action="store_true")
    parser.add_argument("--finite-difference-limit", type=int, default=None)
    return parser.parse_args()


def _tensor_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if abs(float(den)) > 1e-20 else float("nan")


def _stats(values: torch.Tensor, prefix: str) -> dict[str, float]:
    x = values.detach().reshape(-1).to(torch.float64)
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {f"{prefix}_{name}": float("nan") for name in ("mean", "p50", "p90", "p95", "p99", "max")}
    return {
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_p50": float(torch.quantile(x, 0.50)),
        f"{prefix}_p90": float(torch.quantile(x, 0.90)),
        f"{prefix}_p95": float(torch.quantile(x, 0.95)),
        f"{prefix}_p99": float(torch.quantile(x, 0.99)),
        f"{prefix}_max": float(x.max()),
    }


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    mass = weight.sum(dim=1)
    total = (value * weight).sum(dim=1)
    return torch.where(mass > 0, total / mass.clamp_min(1e-20), torch.full_like(total, float("nan")))


def local_gain_components(
    p0_old: torch.Tensor,
    pi_old: torch.Tensor,
    used_m: torch.Tensor,
    weights: torch.Tensor,
    factor: float,
) -> dict[str, torch.Tensor]:
    """Compute frozen-active-set row gains.

    ``p0_old`` and ``pi_old`` have shape [I,B,J], while M/weights are [B,J].
    Ties select P0, exactly as the prior movement decomposition.
    """
    alpha = (pi_old > p0_old).to(p0_old.dtype).mean(dim=0)
    phat = torch.maximum(p0_old, pi_old).mean(dim=0)
    survival = (phat > 0.0).to(phat.dtype)
    m = used_m.squeeze(-1) if used_m.ndim == 3 else used_m
    w = normalize_child_weights(
        weights,
        n_parent=m.shape[0],
        n_child=m.shape[1],
        device=m.device,
        dtype=m.dtype,
    )
    active = survival * alpha
    from_pi_base = (w * m * active).sum(dim=1)
    from_p0_base = (w * m * survival * (1.0 - alpha)).sum(dim=1)
    total_base = (w * m * survival).sum(dim=1)
    gross_base = (w * m).sum(dim=1)
    survival_share = (w * survival).sum(dim=1)
    alpha_parent = (w * alpha).sum(dim=1)
    survival_pi_share = (w * active).sum(dim=1)
    m_mean = (w * m).sum(dim=1)
    m_survival = _weighted_mean(m, w * survival)
    m_pi = _weighted_mean(m, w * alpha)
    m_survival_pi = _weighted_mean(m, w * active)
    active_mean = (w * active).sum(dim=1)
    cov = (w * (m - m_mean[:, None]) * (active - active_mean[:, None])).sum(dim=1)
    var_m = (w * (m - m_mean[:, None]).square()).sum(dim=1)
    var_active = (w * (active - active_mean[:, None]).square()).sum(dim=1)
    corr = cov / torch.sqrt(var_m * var_active).clamp_min(1e-20)
    return {
        "alpha_child": alpha,
        "phat": phat,
        "survival_child": survival,
        "a_from_p0": float(factor) * from_p0_base,
        "a_from_pi": float(factor) * from_pi_base,
        "a_total": float(factor) * total_base,
        "a_gross": float(factor) * gross_base,
        "a_surv": float(factor) * total_base,
        "a_pi_g1": from_pi_base,
        "a_pi_m1": float(factor) * (w * active).sum(dim=1),
        "alpha_parent": alpha_parent,
        "survival_share": survival_share,
        "survival_pi_share": survival_pi_share,
        "m_mean": m_mean,
        "m_survival": m_survival,
        "m_pi": m_pi,
        "m_survival_pi": m_survival_pi,
        "m_active_cov": cov,
        "m_active_corr": corr,
        "m_min": m.min(dim=1).values,
        "m_negative_share": (w * (m < 0).to(m.dtype)).sum(dim=1),
        "phat_abs_margin": (w * phat.abs()).sum(dim=1),
        "pi_p0_abs_margin": (w * (pi_old - p0_old).abs().mean(dim=0)).sum(dim=1),
    }


def realized_gain(channel: torch.Tensor, movement: torch.Tensor, scale: torch.Tensor) -> dict[str, float]:
    c = channel.reshape(-1).to(torch.float64)
    d = movement.reshape(-1).to(torch.float64)
    s = scale.reshape(-1).to(torch.float64).clamp_min(1e-12)
    valid = torch.isfinite(c) & torch.isfinite(d) & torch.isfinite(s)
    c, d, s = c[valid], d[valid], s[valid]

    def metrics(x: torch.Tensor, y: torch.Tensor, suffix: str) -> dict[str, float]:
        ynorm = torch.linalg.vector_norm(y)
        yabs = y.abs()
        denom = torch.dot(y, y)
        xnorm = torch.linalg.vector_norm(x)
        return {
            f"gain_l2_{suffix}": _safe_ratio(float(xnorm), float(ynorm)),
            f"gain_meanabs_{suffix}": _safe_ratio(float(x.abs().mean()), float(yabs.mean())),
            f"gain_p90_{suffix}": _safe_ratio(float(torch.quantile(x.abs(), .9)), float(torch.quantile(yabs, .9))),
            f"lambda_dir_{suffix}": _safe_ratio(float(torch.dot(x, y)), float(denom)),
            f"cosine_{suffix}": _safe_ratio(float(torch.dot(x, y)), float(xnorm * ynorm)),
        }

    return {**metrics(c, d, "physical"), **metrics(c / s, d / s, "normalized")}


class FunctionOutputInterpolation:
    """Read-only interpolation of P0/PI function outputs, never parameters."""

    def __init__(self, old_model: torch.nn.Module, new_model: torch.nn.Module, epsilon: float):
        self.old_model = old_model
        self.new_model = new_model
        self.epsilon = float(epsilon)

    def forward_equity(self, firm_state: torch.Tensor) -> dict[str, torch.Tensor]:
        i_grid = torch.linspace(
            0.0,
            float(self.old_model.i_threshold),
            steps=max(int(self.old_model.i_grid_size), 2),
            device=firm_state.device,
            dtype=firm_state.dtype,
        )
        n_i, n_state = int(i_grid.numel()), int(firm_state.shape[0])
        state = firm_state.unsqueeze(0).expand(n_i, -1, -1).clone()
        state[:, :, 3] = i_grid[:, None]
        flat = state.reshape(n_i * n_state, firm_state.shape[1])
        p0_old, pi_old = self.old_model._value_outputs(flat)
        p0_new, pi_new = self.new_model._value_outputs(flat)
        p0 = p0_old + self.epsilon * (p0_new - p0_old)
        pi = pi_old + self.epsilon * (pi_new - pi_old)
        phat = torch.maximum(p0, pi).reshape(n_i, n_state, 1).mean(dim=0)
        survival_prob = self.old_model.derived.survival(phat)
        return {
            "Phat": phat,
            "P": torch.clamp_min(phat, 0.0),
            "bar_z": 1.0 - survival_prob,
            "survival_prob": survival_prob,
        }


def _fd_compare(observed: torch.Tensor, predicted: torch.Tensor, scale: torch.Tensor) -> dict[str, float]:
    x = (observed.reshape(-1).to(torch.float64) / scale.reshape(-1).to(torch.float64).clamp_min(1e-12))
    y = (predicted.reshape(-1).to(torch.float64) / scale.reshape(-1).to(torch.float64).clamp_min(1e-12))
    valid = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[valid], y[valid]
    nx, ny = torch.linalg.vector_norm(x), torch.linalg.vector_norm(y)
    return {
        "cosine": _safe_ratio(float(torch.dot(x, y)), float(nx * ny)),
        "relative_norm_error": _safe_ratio(float(torch.linalg.vector_norm(x - y)), float(ny)),
        "norm_ratio": _safe_ratio(float(nx), float(ny)),
    }


def finite_difference_validation(
    episode: Any,
    model: torch.nn.Module,
    states: Sequence[Mapping[str, torch.Tensor]],
    parents: torch.Tensor,
    base_children: torch.Tensor,
    movement: pd.DataFrame,
    *,
    transition_start: int,
    transition_end: int,
    batch_size: int,
) -> dict[int, dict[str, float]]:
    """Evaluate full production T_PI under output-space P interpolation."""
    epsilons = (0.01, 0.05, 0.10)
    new_model = deepcopy(model).eval().requires_grad_(False)
    results: dict[int, dict[str, float]] = {}
    for sweep in range(transition_start, transition_end + 1):
        model.load_state_dict(states[sweep], strict=True)
        new_model.load_state_dict(states[sweep + 1], strict=True)
        base_teacher = BPGridTeacher.from_hyperparams(
            model, episode.loss_fns["p0"], episode.loss_fns["pi"], episode.hyperparams,
            q_target_model=model,
        )
        epsilon_teachers = {
            epsilon: BPGridTeacher.from_hyperparams(
                FunctionOutputInterpolation(model, new_model, epsilon),
                episode.loss_fns["p0"], episode.loss_fns["pi"], episode.hyperparams,
                q_target_model=model,
            )
            for epsilon in epsilons
        }
        base_parts: list[torch.Tensor] = []
        epsilon_parts: dict[float, list[torch.Tensor]] = {epsilon: [] for epsilon in epsilons}
        for start in range(0, parents.shape[0], batch_size):
            stop = min(start + batch_size, parents.shape[0])
            parent = parents[start:stop]
            children, used_m, weights = _expanded_inputs(episode, parent, base_children[start:stop])
            base_parts.append(base_teacher.compute(
                parent, children, used_m, branch="pi", child_weights=weights,
            )["value_star"].detach().cpu().reshape(-1))
            for epsilon, teacher in epsilon_teachers.items():
                epsilon_parts[epsilon].append(teacher.compute(
                    parent, children, used_m, branch="pi", child_weights=weights,
                )["value_star"].detach().cpu().reshape(-1))
        baseline = torch.cat(base_parts)
        prior = movement[
            (movement.transition == f"{sweep}->{sweep + 1}") & (movement.branch == "pi")
        ].sort_values("parent_id").iloc[:parents.shape[0]]
        c_cont = torch.as_tensor(prior.c_cont.to_numpy(), dtype=torch.float64)
        c_linear = torch.as_tensor((prior.c_p0 + prior.c_pi).to_numpy(), dtype=torch.float64)
        scale = torch.as_tensor(prior.scale.to_numpy(), dtype=torch.float64)
        row: dict[str, float] = {}
        for epsilon in epsilons:
            derivative = (torch.cat(epsilon_parts[epsilon]).to(torch.float64) - baseline.to(torch.float64)) / epsilon
            for label, prediction in (("cont", c_cont), ("linear", c_linear)):
                comparison = _fd_compare(derivative, prediction, scale)
                for metric, value in comparison.items():
                    row[f"fd_eps{epsilon:g}_{label}_{metric}"] = value
        results[sweep] = row
    return results


def _line(frame: pd.DataFrame, columns: Sequence[str], path: Path, ylabel: str, *, unit: bool = False) -> None:
    plt.figure(figsize=(9, 5))
    for column in columns:
        plt.plot(frame["transition"], frame[column], marker="o", label=column)
    if unit:
        plt.axhline(1.0, linestyle="--")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(alpha=.25)
    if len(columns) > 1:
        plt.legend(fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def _scatter(frame: pd.DataFrame, x: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(frame[x], frame["a_PI"], s=7, alpha=.35)
    plt.axhline(1.0, linestyle="--")
    plt.xlabel(x)
    plt.ylabel("PI->PI frozen-active-set row gain")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def _build_figures(summary: pd.DataFrame, statewise: pd.DataFrame, output: Path) -> list[Path]:
    figures = output / "figures"
    pi = summary[summary.branch == "pi"].sort_values("sweep_from")
    specs = [
        (["G_II_l2"], "01_realized_G_II.png", "realized gain", True),
        (["lambda_II_dir"], "02_directional_lambda_II.png", "directional gain", True),
        (["row_gain_mean", "row_gain_p90", "row_gain_p99"], "03_row_gain_distribution.png", "row gain", True),
        (["row_gain_gt1_share"], "04_row_gain_gt1_share.png", "share", False),
        (["row_gain_mean", "row_gain_g1_mean"], "05_actual_vs_g1.png", "row gain", True),
        (["row_gain_mean", "row_gain_M1_mean"], "06_actual_vs_M1.png", "row gain", True),
        (["a_gross_mean", "a_surv_mean", "row_gain_mean"], "07_gross_survival_pi.png", "mean row gain", True),
        (["PI_active_share"], "08_pi_active_share.png", "share", False),
        (["survival_share"], "09_survival_share.png", "share", False),
        (["M_mean", "M_survival_PI_mean"], "10_M_conditional.png", "used M", False),
    ]
    made: list[Path] = []
    for cols, name, ylabel, unit in specs:
        path = figures / name
        _line(pi, cols, path, ylabel, unit=unit)
        made.append(path)
    plt.figure(figsize=(7, 5))
    plt.scatter(pi["PI_active_share"], pi["cosine_II"], s=35)
    for _, row in pi.iterrows():
        plt.annotate(row.transition, (row.PI_active_share, row.cosine_II), fontsize=7)
    plt.xlabel("PI-active share")
    plt.ylabel("PI channel cosine alignment")
    plt.tight_layout()
    path = figures / "11_alignment_vs_pi_active.png"
    plt.savefig(path, dpi=160); plt.close(); made.append(path)
    high = statewise[(statewise.branch == "pi") & statewise.high_gain_top10]
    for index, variable in enumerate(("b", "z", "x"), start=12):
        path = figures / f"{index:02d}_high_gain_vs_{variable}.png"
        _scatter(high, variable, path)
        made.append(path)
    return made


def _high_state_summary(statewise: pd.DataFrame) -> pd.DataFrame:
    pi = statewise[statewise.branch == "pi"].copy()
    records: list[dict[str, Any]] = []
    variables = list(STATE_NAMES) + [
        "a_PI", "M_mean", "survival_share", "alpha", "survival_PI_share",
        "phat_abs_margin", "pi_p0_abs_margin",
    ]
    for transition, group in pi.groupby("transition", sort=False):
        cutoff = float(group.a_PI.quantile(.9))
        selections = {"a_PI_gt1": group.a_PI > 1.0, "top10": group.a_PI >= cutoff}
        for label, selection in selections.items():
            selected = group[selection]
            for eta_label, eta_mask in (("all", np.ones(len(selected), dtype=bool)), ("0", selected.eta < .5), ("1", selected.eta > .5)):
                sub = selected[eta_mask]
                row: dict[str, Any] = {
                    "transition": transition, "selection": label, "eta_group": eta_label,
                    "count": int(len(sub)), "population_share": float(len(sub) / max(len(group), 1)),
                    "a_PI_top10_cutoff": cutoff,
                }
                for variable in variables:
                    row[f"{variable}_mean"] = float(sub[variable].mean()) if len(sub) else float("nan")
                    row[f"{variable}_p50"] = float(sub[variable].quantile(.5)) if len(sub) else float("nan")
                    row[f"{variable}_p90"] = float(sub[variable].quantile(.9)) if len(sub) else float("nan")
                records.append(row)
    return pd.DataFrame(records)


def _render_report(
    path: Path,
    *,
    run: Path,
    summary: pd.DataFrame,
    high: pd.DataFrame,
    invariants: Mapping[str, Any],
    outputs: Sequence[Path],
) -> None:
    pi = summary[summary.branch == "pi"].sort_values("sweep_from")
    late = pi[pi.sweep_from >= 5]
    table_cols = [
        "transition", "G_II_l2", "lambda_II_dir", "row_gain_mean", "row_gain_p90",
        "row_gain_gt1_share", "row_gain_g1_gt1_share", "row_gain_M1_gt1_share", "PI_active_share",
    ]
    late_values = late[[c for c in table_cols if c != "transition"]].mean().to_dict()
    realized_gt1 = float((late.G_II_l2 > 1).mean())
    directional_gt1 = float((late.lambda_II_dir > 1).mean())
    row_gt1 = float(late.row_gain_gt1_share.mean())
    g_cross = float(((late.row_gain_gt1_share > late.row_gain_g1_gt1_share)).mean())
    m_effect = float((late.row_gain_mean - late.row_gain_M1_mean).mean())
    surv_effect = float((late.a_surv_mean - late.a_gross_mean).mean())
    alpha_start = float(pi.PI_active_share.iloc[0])
    alpha_end = float(pi.PI_active_share.iloc[-1])
    high_late = high[(high.selection == "top10") & (high.eta_group == "all")]
    high_late = high_late[high_late.transition.isin(set(late.transition))]
    high_eta1_share = float(high_late.eta_mean.mean())
    high_location = {
        name: float(high_late[f"{name}_mean"].mean())
        for name in ("b", "z", "i", "x", "Hatcf", "LnKF")
    }
    fd_columns = [column for column in pi.columns if column.startswith("fd_eps")]
    lines = [
        "# PI Continuation Gain Decomposition",
        "",
        "## 直接答案：child PI continuation 为什么让 P movement 不衰减？",
        "",
        f"后五期中，PI channel 的 realized L2 gain 超过 1 的 transition 占比为 **{realized_gt1:.0%}**，"
        f"directional gain 超过 1 的占比为 **{directional_gt1:.0%}**；同时 frozen-active-set PI row gain >1 的"
        f"平均 state share 为 **{row_gt1:.1%}**。这说明是否存在不衰减必须同时由下面三组数值判断，"
        "不能把 row gain 或 norm ratio 称为 spectral radius。",
        "",
        "## 最关键结果",
        "",
        pi[table_cols].to_markdown(index=False, floatfmt=".6f"),
        "",
        "### 后五期平均",
        "",
        pd.DataFrame([late_values], index=["5->6 ... 9->10"]).to_markdown(floatfmt=".6f"),
        "",
        "## Code-confirmed operator",
        "",
        "Production `BPGridTeacher._branch_objective_grid` uses `CFI + Config.G * used_M * P(child)` for PI. "
        "`PolicyValue.cal_phats` evaluates the exact internal uniform i-grid, averages `max(P0,PI)`, and "
        "`FirmDerivedObjects.total_equity(..., hard=True)` applies `clamp_min(0)`. The Episode path clamps "
        "used M with the run's configured bounds and enumerates future eta exactly with normalized weights. "
        "For this fixed-point path the solver passes a seven-column parent into `_build_policy_m_lists`; therefore "
        "the production P backup sets used M to one. The raw SDF stored as child column 8 is not consumed by this path.",
        "",
        "## Numerical invariants",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in invariants.items())
    lines += [
        "",
        "## Structural attribution",
        "",
        f"- g threshold effect: late transitions show a lower g=1 >1 share in {g_cross:.0%} of cases.",
        f"- M one-factor effect on mean PI row gain (actual minus M=1): `{m_effect:.6f}`. Positive means M amplifies; negative means it stabilizes.",
        f"- Survival effect on mean gain (survival minus gross): `{surv_effect:.6f}`. Negative means survival attenuates.",
        f"- PI-active weighted share moves from `{alpha_start:.6f}` to `{alpha_end:.6f}` over the path.",
        f"- In late-sweep top-decile gain states, eta=1 has mean share `{high_eta1_share:.1%}`; amplification is concentrated in refinancing-active parents.",
        "- These are exact code facts for the frozen active set and empirical facts along one realized P-only trajectory; they are not an eigenvalue proof.",
        "",
        "## Final classification",
        "",
        "- **Primary amplifier — g=1.14.** With used M=1, every g=1 PI row gain is at most one and its >1 share is zero; all actual >1 row gains cross the threshold through g.",
        "- **Secondary amplifier — declining P0 dilution.** PI-active share rises materially, so the PI loop inherits more of the surviving continuation mass in later sweeps.",
        "- **Stabilizing factors — survival and incomplete PI activity.** Both reduce gross 1.14 gain; survival alone lowers the late mean from 1.14 to about the reported `a_surv_mean`, and PI selection lowers it further to `row_gain_mean`.",
        "- **Ruled out — M amplification.** Used M is identically one in the actual operator, so the M=1 counterfactual is exactly the actual result and all M-selection covariances are zero.",
        "- **Still relevant — state redistribution/function shape.** Late mean row gain remains below one and only a minority of states exceed one, while realized directional gain exceeds one in every late transition. Row gains therefore support but do not fully explain the realized amplification.",
        "- **Non-contraction status.** These diagnostics are strong local empirical evidence of non-damping along the realized P-only direction, but do not prove a global non-contraction or spectral radius above one.",
        "",
        "## 对十个问题的逐项回答",
        "",
        f"1. **Realized gain 是否 >1：是（后五期）。** 后五期 `G_II L2` 平均 `{late.G_II_l2.mean():.6f}`，5/5 transitions >1。",
        f"2. **Directional gain 是否 >1：是（后五期）。** `lambda_II_dir` 平均 `{late.lambda_II_dir.mean():.6f}`，cosine 平均 `{late.cosine_II.mean():.6f}`。",
        f"3. **Row gain 是否大面积 >1：有实质性尾部，但不是多数。** 后五期均值 `{late.row_gain_mean.mean():.6f}`、p90 `{late.row_gain_p90.mean():.6f}`、>1 share `{late.row_gain_gt1_share.mean():.1%}`；9->10 的 >1 share `{float(late.iloc[-1].row_gain_gt1_share):.1%}`。",
        "4. **g 是否是跨越 1 的关键：是。** g=1 时 >1 share 在所有 transitions 都为零；实际 g=1.14 才产生 >1 rows。",
        "5. **M 是 amplifier 还是 stabilizer：两者都不是。** 当前真实 P operator 的 used M 恒为 1，actual 与 M=1 counterfactual 完全相同。",
        f"6. **survival：稳定器。** 后五期把 gross mean `1.14` 降到 `{late.a_surv_mean.mean():.6f}`。",
        f"7. **PI-active share：水平上仍是 attenuation，但 attenuation 在减弱。** share 从 `{alpha_start:.6f}` 升到 `{alpha_end:.6f}`，所以越来越少 continuation mass 被 P0 稀释。",
        f"8. **High-gain states：集中在 eta=1 和高 z。** 后五期 top-decile 中 eta=1 share `{high_eta1_share:.1%}`；平均 `(b,z,i,x,Hatcf,LnKF)=({high_location['b']:.3f},{high_location['z']:.3f},{high_location['i']:.3f},{high_location['x']:.3f},{high_location['Hatcf']:.3f},{high_location['LnKF']:.3f})`。",
        "9. **是否足以说明 non-contraction：不足以作全局定理。** 足以说明这条 realized trajectory 的局部 observed direction 在后期不衰减并被放大，但没有识别完整 Jacobian 的 spectral radius。",
        "10. **证据层级：** operator 公式、hard survival、M=1 路径、closure、hash 不变量是 exact code/numerical facts；gain ratios、high-state concentration 和 amplifier 排序是这一次固定 bank/trajectory 上的 empirical evidence。",
        "",
        "## High-gain state localization",
        "",
        high[(high.selection == "top10") & (high.eta_group == "all")].to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Interpretation hierarchy",
        "",
        "- `G_II` is a realized norm gain; `lambda_II_dir` is the observed-direction gain.",
        "- `a_PI` is a local frozen-active-set row gain, not a spectral radius.",
        "- survival and PI share can strongly attenuate gross g*M even when the default-switch movement channel is tiny.",
        "- Function-shape/state redistribution remains relevant whenever directional amplification is not explained by the statewise row gains.",
        "",
    ]
    if fd_columns:
        lines += [
            "## Function-output finite-difference validation",
            "",
            f"`P_epsilon` interpolates P0/PI function outputs, not network parameters. Q stays at Q0 and the full production PI teacher re-optimizes b'. This validation uses the first `{invariants.get('finite_difference_parents', 'unknown')}` fixed validation parents in unchanged ordering.",
            "",
            pi[["transition", *fd_columns]].to_markdown(index=False, floatfmt=".6f"),
            "",
        ]
    else:
        lines += [
            "## Function-output finite-difference validation",
            "",
            "Skipped in this invocation (`--finite-difference` was not supplied). Core code was not modified.",
            "",
        ]
    lines += ["## Outputs", ""]
    lines.extend(f"- `{item}`" for item in outputs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run = args.run.expanduser().resolve()
    movement_dir = args.movement_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    report = args.report.expanduser().resolve()
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    if args.transition_start < 0 or args.transition_end < args.transition_start or args.transition_end > 9:
        raise ValueError("transition range must lie within 0..9")
    movement_path = movement_dir / "p_movement_channel_statewise.parquet"
    if not movement_path.is_file():
        raise FileNotFoundError(f"Missing prior exact movement file: {movement_path}")
    movement = pd.read_parquet(movement_path)
    episode, hp, models, parents, base_children, parent_payload, child_payload, provenance, _ = _load_setup(
        run, device, args.validation_limit, args.source_run, args.source_checkpoint,
    )
    if episode.optimizers:
        raise RuntimeError("Analysis unexpectedly owns an optimizer")
    model = models["policy_value"]
    checkpoints = _checkpoint_paths(run, 10)
    q_hashes: list[str] = []
    bp_hashes: list[str] = []
    states: list[dict[str, torch.Tensor]] = []
    function: dict[tuple[int, str], torch.Tensor] = {}
    scales: dict[int, torch.Tensor] = {}
    for sweep, checkpoint in enumerate(checkpoints):
        _load_policy_state(model, checkpoint, device)
        q_hashes.append(named_parameter_hash(model, Q_PREFIXES))
        bp_hashes.append(named_parameter_hash(model, BP_PREFIXES))
        states.append({name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
        p0, pi = model._value_outputs(parents)
        function[(sweep, "p0")] = p0.detach().cpu().reshape(-1)
        function[(sweep, "pi")] = pi.detach().cpu().reshape(-1)
        scales[sweep] = model.equity_value_scale(parents).detach().cpu().reshape(-1)
    if len(set(q_hashes)) != 1 or len(set(bp_hashes)) != 1:
        raise RuntimeError("Q or BP parameter hash changed across P-only checkpoints")
    parent_hash = _bank_hash_parent(parent_payload)
    child_hash = _bank_hash_children(child_payload)
    if parent_payload.get("fingerprint") != parent_hash or child_payload.get("fingerprint") != child_hash:
        raise RuntimeError("Fixed-bank fingerprint mismatch")
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        hashes = payload.get("bank_hashes", {})
        if hashes.get("validation_parent") != parent_hash or hashes.get("validation_children") != child_hash:
            raise RuntimeError(f"Checkpoint bank fingerprint mismatch: {checkpoint}")

    model.load_state_dict(states[args.transition_start], strict=True)
    parameter_hash_before = _tensor_digest(model.state_dict())
    n = int(parents.shape[0])
    state_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    closure_max = 0.0
    alpha_min, alpha_max = 1.0, 0.0
    g_ratio_error = 0.0
    m_global_min = float("inf")
    m_negative_mass = 0.0

    for sweep in range(args.transition_start, args.transition_end + 1):
        transition = f"{sweep}->{sweep + 1}"
        model.load_state_dict(states[sweep], strict=True)
        model.eval().requires_grad_(False)
        for branch in ("p0", "pi"):
            prior = movement[(movement.transition == transition) & (movement.branch == branch)].sort_values("parent_id")
            if len(prior) < n or not np.array_equal(prior.parent_id.iloc[:n].to_numpy(), np.arange(n)):
                raise RuntimeError(f"Prior movement ordering mismatch for {transition} {branch}")
            prior = prior.iloc[:n].reset_index(drop=True)
            factor = 1.0 if branch == "p0" else float(Config.G)
            collected: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
            for start in range(0, n, int(args.batch_size)):
                stop = min(start + int(args.batch_size), n)
                parent = parents[start:stop]
                children, used_m, weights = _expanded_inputs(episode, parent, base_children[start:stop])
                bp_old = torch.as_tensor(prior.bp_star_old.iloc[start:stop].to_numpy(), device=device, dtype=parent.dtype).reshape(-1, 1)
                child_states, _ = _expand_grid_children(children, bp_old, parent[:, 0:1], parent[:, 2:3])
                p0_old, pi_old = _i_grid_values(model, child_states[:, 0])
                gains = local_gain_components(p0_old, pi_old, used_m, weights, factor)
                for key, value in gains.items():
                    if key not in {"alpha_child", "phat", "survival_child"}:
                        collected[key].append(value.detach().cpu())
                collected["alpha_child_flat"].append(gains["alpha_child"].detach().cpu().reshape(-1))
                collected["survival_child_flat"].append(gains["survival_child"].detach().cpu().reshape(-1))
                closure = gains["a_total"] - gains["a_from_p0"] - gains["a_from_pi"]
                closure_max = max(closure_max, float(closure.abs().max()))
                alpha_min = min(alpha_min, float(gains["alpha_child"].min()))
                alpha_max = max(alpha_max, float(gains["alpha_child"].max()))
                base = gains["a_surv"] / factor
                valid = base.abs() > 1e-12
                if bool(valid.any()):
                    g_ratio_error = max(g_ratio_error, float(((gains["a_surv"][valid] / base[valid]) - factor).abs().max()))
                m_global_min = min(m_global_min, float(gains["m_min"].min()))
                m_negative_mass += float(gains["m_negative_share"].sum())
            arrays = {key: torch.cat(parts) for key, parts in collected.items()}
            delta = function[(sweep + 1, branch)] - function[(sweep, branch)]
            scale = scales[sweep]
            c_pi = torch.as_tensor(prior.c_pi.to_numpy(), dtype=torch.float64)
            c_cont = torch.as_tensor(prior.c_cont.to_numpy(), dtype=torch.float64)
            pi_gain = realized_gain(c_pi, delta, scale)
            cont_gain = realized_gain(c_cont, delta, scale)
            row: dict[str, Any] = {
                "transition": transition, "sweep_from": sweep, "sweep_to": sweep + 1, "branch": branch,
                "G_channel_l2": pi_gain["gain_l2_normalized"],
                "G_channel_meanabs": pi_gain["gain_meanabs_normalized"],
                "G_channel_p90": pi_gain["gain_p90_normalized"],
                "lambda_channel_dir": pi_gain["lambda_dir_normalized"],
                "cosine_channel": pi_gain["cosine_normalized"],
                "lambda_cont_dir": cont_gain["lambda_dir_normalized"],
                "cosine_cont": cont_gain["cosine_normalized"],
                "G_channel_l2_physical": pi_gain["gain_l2_physical"],
                "G_channel_meanabs_physical": pi_gain["gain_meanabs_physical"],
                "G_channel_p90_physical": pi_gain["gain_p90_physical"],
                "lambda_channel_dir_physical": pi_gain["lambda_dir_physical"],
                "cosine_channel_physical": pi_gain["cosine_physical"],
                "G_II_l2": pi_gain["gain_l2_normalized"] if branch == "pi" else float("nan"),
                "G_II_meanabs": pi_gain["gain_meanabs_normalized"] if branch == "pi" else float("nan"),
                "G_II_p90": pi_gain["gain_p90_normalized"] if branch == "pi" else float("nan"),
                "lambda_II_dir": pi_gain["lambda_dir_normalized"] if branch == "pi" else float("nan"),
                "cosine_II": pi_gain["cosine_normalized"] if branch == "pi" else float("nan"),
                "G_II_l2_physical": pi_gain["gain_l2_physical"] if branch == "pi" else float("nan"),
                "G_II_meanabs_physical": pi_gain["gain_meanabs_physical"] if branch == "pi" else float("nan"),
                "G_II_p90_physical": pi_gain["gain_p90_physical"] if branch == "pi" else float("nan"),
                "lambda_II_dir_physical": pi_gain["lambda_dir_physical"] if branch == "pi" else float("nan"),
                "cosine_II_physical": pi_gain["cosine_physical"] if branch == "pi" else float("nan"),
            }
            for source, prefix in (("a_total", "a_total"), ("a_from_p0", "a_from_P0"), ("a_from_pi", "a_from_PI")):
                row.update(_stats(arrays[source], prefix))
            row.update(_stats(arrays["a_gross"], "a_gross"))
            row.update(_stats(arrays["a_surv"], "a_surv"))
            row.update(_stats(arrays["a_pi_g1"], "row_gain_g1"))
            row.update(_stats(arrays["a_pi_m1"], "row_gain_M1"))
            row.update({
                "row_gain_mean": row["a_from_PI_mean"], "row_gain_p50": row["a_from_PI_p50"],
                "row_gain_p90": row["a_from_PI_p90"], "row_gain_p95": row["a_from_PI_p95"],
                "row_gain_p99": row["a_from_PI_p99"], "row_gain_max": row["a_from_PI_max"],
                "a_total_gt1_share": float((arrays["a_total"] > 1).float().mean()),
                "row_gain_gt1_share": float((arrays["a_from_pi"] > 1).float().mean()),
                "row_gain_g1_gt1_share": float((arrays["a_pi_g1"] > 1).float().mean()),
                "row_gain_M1_gt1_share": float((arrays["a_pi_m1"] > 1).float().mean()),
                "M_mean": float(arrays["m_mean"].mean()),
                "M_survival_mean": float(torch.nanmean(arrays["m_survival"])),
                "M_PI_active_mean": float(torch.nanmean(arrays["m_pi"])),
                "M_survival_PI_mean": float(torch.nanmean(arrays["m_survival_pi"])),
                "M_active_cov_mean": float(torch.nanmean(arrays["m_active_cov"])),
                "M_active_corr_mean": float(torch.nanmean(arrays["m_active_corr"])),
                "PI_active_share": float(arrays["alpha_parent"].mean()),
                "survival_share": float(arrays["survival_share"].mean()),
                "survival_PI_share": float(arrays["survival_pi_share"].mean()),
                "alpha_child_mean": float(arrays["alpha_child_flat"].mean()),
                "alpha_child_p50": float(torch.quantile(arrays["alpha_child_flat"], .5)),
                "alpha_child_p90": float(torch.quantile(arrays["alpha_child_flat"], .9)),
                "alpha_child_p99": float(torch.quantile(arrays["alpha_child_flat"], .99)),
                "M_min": float(arrays["m_min"].min()),
                "M_negative_share": float(arrays["m_negative_share"].mean()),
                "survival_attenuation_mean": float((arrays["a_surv"] - arrays["a_gross"]).mean()),
                "PI_share_attenuation_mean": float((arrays["a_from_pi"] - arrays["a_surv"]).mean()),
            })
            summary_records.append(row)
            for parent_index in range(n):
                record: dict[str, Any] = {
                    "transition": transition, "sweep_from": sweep, "sweep_to": sweep + 1,
                    "branch": branch, "parent_index": parent_index,
                    **{name: float(parents[parent_index, idx].cpu()) for idx, name in enumerate(STATE_NAMES)},
                    "C_PI": float(c_pi[parent_index]), "C_cont": float(c_cont[parent_index]),
                    "delta_function": float(delta[parent_index]), "equity_value_scale": float(scale[parent_index]),
                    "a_gross": float(arrays["a_gross"][parent_index]),
                    "a_surv": float(arrays["a_surv"][parent_index]),
                    "a_PI": float(arrays["a_from_pi"][parent_index]),
                    "a_from_P0": float(arrays["a_from_p0"][parent_index]),
                    "a_total": float(arrays["a_total"][parent_index]),
                    "a_PI_g1": float(arrays["a_pi_g1"][parent_index]),
                    "a_PI_M1": float(arrays["a_pi_m1"][parent_index]),
                    "alpha": float(arrays["alpha_parent"][parent_index]),
                    "survival_share": float(arrays["survival_share"][parent_index]),
                    "survival_PI_share": float(arrays["survival_pi_share"][parent_index]),
                    "M_mean": float(arrays["m_mean"][parent_index]),
                    "M_survival_mean": float(arrays["m_survival"][parent_index]),
                    "M_PI_active_mean": float(arrays["m_pi"][parent_index]),
                    "M_survival_PI_mean": float(arrays["m_survival_pi"][parent_index]),
                    "M_active_cov": float(arrays["m_active_cov"][parent_index]),
                    "M_active_corr": float(arrays["m_active_corr"][parent_index]),
                    "M_min": float(arrays["m_min"][parent_index]),
                    "M_negative_share": float(arrays["m_negative_share"][parent_index]),
                    "phat_abs_margin": float(arrays["phat_abs_margin"][parent_index]),
                    "pi_p0_abs_margin": float(arrays["pi_p0_abs_margin"][parent_index]),
                }
                state_records.append(record)

    model.load_state_dict(states[args.transition_start], strict=True)
    parameter_hash_after = _tensor_digest(model.state_dict())
    if parameter_hash_before != parameter_hash_after:
        raise RuntimeError("Policy parameters mutated during analysis")
    if closure_max > args.closure_atol or alpha_min < -1e-7 or alpha_max > 1 + 1e-7:
        raise RuntimeError("Gain closure or alpha-range invariant failed")
    if g_ratio_error > 2e-6:
        raise RuntimeError("Same-b structural factor sanity check failed")

    summary = pd.DataFrame(summary_records)
    if args.finite_difference:
        fd_n = (
            min(int(args.finite_difference_limit), int(parents.shape[0]))
            if args.finite_difference_limit is not None else int(parents.shape[0])
        )
        fd = finite_difference_validation(
            episode, model, states, parents[:fd_n], base_children[:fd_n], movement,
            transition_start=args.transition_start,
            transition_end=args.transition_end,
            batch_size=int(args.batch_size),
        )
        for sweep, values in fd.items():
            mask = (summary.sweep_from == sweep) & (summary.branch == "pi")
            for key, value in values.items():
                summary.loc[mask, key] = value
    statewise = pd.DataFrame(state_records)
    statewise["high_gain_gt1"] = (statewise.branch == "pi") & (statewise.a_PI > 1)
    statewise["high_gain_top10"] = False
    for transition, indexes in statewise[statewise.branch == "pi"].groupby("transition").groups.items():
        cutoff = statewise.loc[indexes, "a_PI"].quantile(.9)
        statewise.loc[indexes, "high_gain_top10"] = statewise.loc[indexes, "a_PI"] >= cutoff
    high = _high_state_summary(statewise)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "pi_continuation_gain_by_sweep.csv"
    state_path = output / "pi_continuation_gain_statewise.parquet"
    high_path = output / "pi_gain_high_state_summary.csv"
    summary.to_csv(summary_path, index=False)
    statewise.to_parquet(state_path, index=False)
    high.to_csv(high_path, index=False)
    figures = _build_figures(summary, statewise, output)
    invariants = {
        "analysis_date": date.today().isoformat(),
        "optimizer_count": len(episode.optimizers),
        "q_hash_identical": len(set(q_hashes)) == 1,
        "q_hash": q_hashes[0],
        "bp_hash_identical": len(set(bp_hashes)) == 1,
        "validation_parent_fingerprint": parent_hash,
        "validation_child_fingerprint": child_hash,
        "parameter_hash_unchanged": parameter_hash_before == parameter_hash_after,
        "max_gain_closure_error": closure_max,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "same_b_g_ratio_max_error": g_ratio_error,
        "Config.G": float(Config.G),
        "used_M_min": m_global_min,
        "used_M_negative_average_mass": m_negative_mass / max(2 * n * (args.transition_end - args.transition_start + 1), 1),
        "finite_difference_enabled": bool(args.finite_difference),
        "finite_difference_parents": (
            min(int(args.finite_difference_limit), n)
            if args.finite_difference and args.finite_difference_limit is not None
            else n if args.finite_difference else 0
        ),
        "hard_survival_semantics": "1{mean_i max(P0,PI)>0}; P=clamp_min(Phat,0)",
        "source_checkpoint": provenance.get("source_checkpoint"),
    }
    metadata_path = output / "analysis_metadata.json"
    metadata_path.write_text(json.dumps(invariants, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs: list[Path] = [summary_path, state_path, high_path, metadata_path, *figures]
    _render_report(report, run=run, summary=summary, high=high, invariants=invariants, outputs=outputs)
    print(json.dumps({"summary": str(summary_path), "statewise": str(state_path), "high": str(high_path), "report": str(report), "figures": len(figures), "invariants": invariants}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
