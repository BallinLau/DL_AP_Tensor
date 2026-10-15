#!/usr/bin/env python3
"""Post-hoc decomposition of P-only Bellman-target movement.

This program is intentionally read-only with respect to training.  It loads the
completed P-only sweep checkpoints and the run's immutable validation banks,
reuses the production BPGridTeacher, and never constructs or steps an optimizer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.economic_config import AnalysisEconomicConfig  # noqa: E402
from config import Config, HyperParams  # noqa: E402
from experiments.run_utils import load_training_warmstart  # noqa: E402
from training.bp_grid_teacher import (  # noqa: E402
    BPGridTeacher,
    _expand_grid_children,
)
from training.episode import Episode  # noqa: E402
from training.inner_fixed_point import (  # noqa: E402
    InnerFPChildBank,
    InnerFPCollocationBank,
    module_hash,
    named_parameter_hash,
)
from utils.firm_transition import normalize_child_weights  # noqa: E402


P_PREFIXES = ("value_encoder", "v0_head", "vi_head", "barz_model", "bari_model")
Q_PREFIXES = ("q_encoder", "q_head")
BP_PREFIXES = ("policy_encoder", "bp0_head", "bpi_head")
CHANNELS = ("c_p0", "c_pi", "c_inv", "c_default", "c_bp")
CHANNEL_LABELS = {
    "c_p0": "child P0",
    "c_pi": "child PI",
    "c_inv": "investment switch",
    "c_default": "default switch",
    "c_bp": "BP/search grid",
}
FIXED_DEFAULT_BUCKETS = (0.0, 0.01, 0.05, 0.20, math.inf)
FIXED_INVEST_BUCKETS = (0.0, 0.01, 0.05, 0.20, math.inf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--common-grid-size", type=int, default=101)
    parser.add_argument("--max-transitions", type=int, default=10)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--closure-atol", type=float, default=2e-5)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def _restore_hyperparams(values: Mapping[str, Any]) -> HyperParams:
    hp = HyperParams()
    for key, value in values.items():
        setattr(hp, key, value)
    return hp


def _tensor_hash(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = sha256()
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _bank_hash_parent(payload: Mapping[str, Any]) -> str:
    return _tensor_hash((
        ("parent_states", payload["parent_states"]),
        ("source_kind", payload["source_kind"]),
    ))


def _bank_hash_children(payload: Mapping[str, Any]) -> str:
    return _tensor_hash(tuple((name, payload[name]) for name in (
        "children", "eps_x", "eps_z", "u_eta", "u_i"
    )))


def _finite(v: torch.Tensor) -> torch.Tensor:
    flat = v.detach().reshape(-1).to(torch.float64)
    return flat[torch.isfinite(flat)]


def _stats(v: torch.Tensor, prefix: str, *, signed: bool = True) -> dict[str, float]:
    flat = _finite(v)
    out: dict[str, float] = {}
    if flat.numel() == 0:
        keys = ("signed_mean", "abs_mean", "abs_p50", "abs_p90", "abs_p99", "abs_max")
        return {f"{prefix}_{key}": float("nan") for key in keys}
    av = flat.abs()
    out[f"{prefix}_signed_mean"] = float(flat.mean()) if signed else float("nan")
    out[f"{prefix}_abs_mean"] = float(av.mean())
    out[f"{prefix}_abs_p50"] = float(torch.quantile(av, 0.50))
    out[f"{prefix}_abs_p90"] = float(torch.quantile(av, 0.90))
    out[f"{prefix}_abs_p99"] = float(torch.quantile(av, 0.99))
    out[f"{prefix}_abs_max"] = float(av.max())
    return out


def _alignment(channel: torch.Tensor, total: torch.Tensor, scale: torch.Tensor) -> float:
    c = (channel / scale.clamp_min(1e-8)).reshape(-1).to(torch.float64)
    y = (total / scale.clamp_min(1e-8)).reshape(-1).to(torch.float64)
    valid = torch.isfinite(c) & torch.isfinite(y)
    denominator = torch.dot(y[valid], y[valid])
    if denominator <= 1e-20:
        return float("nan")
    return float(torch.dot(c[valid], y[valid]) / denominator)


def _ratio(num: torch.Tensor, den: torch.Tensor, kind: str) -> float:
    a, b = _finite(num), _finite(den)
    if kind == "l2":
        denominator = torch.linalg.vector_norm(b)
        return float(torch.linalg.vector_norm(a) / denominator) if denominator > 1e-20 else float("nan")
    if kind == "mean_abs":
        denominator = b.abs().mean()
        return float(a.abs().mean() / denominator) if denominator > 1e-20 else float("nan")
    if kind == "p90":
        denominator = torch.quantile(b.abs(), 0.90)
        return float(torch.quantile(a.abs(), 0.90) / denominator) if denominator > 1e-20 else float("nan")
    raise ValueError(kind)


def child_equity_decomposition(
    p0_old: torch.Tensor,
    pi_old: torch.Tensor,
    p0_new: torch.Tensor,
    pi_new: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Exact hard-equity decomposition over the internal investment grid.

    Inputs have shape [I, B, J] (a final singleton dimension is also accepted).
    Ties deterministically use P0, matching torch.max values while leaving the
    exact value decomposition unchanged.
    """
    p0_old, pi_old = p0_old.squeeze(-1), pi_old.squeeze(-1)
    p0_new, pi_new = p0_new.squeeze(-1), pi_new.squeeze(-1)
    old_i = pi_old > p0_old
    new_i = pi_new > p0_new
    old_selected = torch.where(old_i, pi_old, p0_old)
    new_old_i = torch.where(old_i, pi_new, p0_new)
    new_selected = torch.maximum(p0_new, pi_new)
    phat_old = old_selected.mean(dim=0)
    phat_new_old_i = new_old_i.mean(dim=0)
    phat_new = new_selected.mean(dim=0)
    c_p0_raw = ((~old_i).to(p0_old.dtype) * (p0_new - p0_old)).mean(dim=0)
    c_pi_raw = (old_i.to(p0_old.dtype) * (pi_new - pi_old)).mean(dim=0)
    c_inv_i = new_selected - new_old_i
    c_inv_raw = c_inv_i.mean(dim=0)
    survive_old = (phat_old > 0.0).to(phat_old.dtype)
    survive_new = (phat_new > 0.0).to(phat_new.dtype)
    c_p0 = survive_old * c_p0_raw
    c_pi = survive_old * c_pi_raw
    c_inv = survive_old * c_inv_raw
    c_default = (survive_new - survive_old) * phat_new
    p_old = torch.relu(phat_old)
    p_new = torch.relu(phat_new)
    return {
        "old_i": old_i,
        "new_i": new_i,
        "phat_old": phat_old,
        "phat_new": phat_new,
        "phat_new_old_i": phat_new_old_i,
        "p_old": p_old,
        "p_new": p_new,
        "c_p0_raw": c_p0_raw,
        "c_pi_raw": c_pi_raw,
        "c_inv_raw": c_inv_raw,
        "c_inv_i": c_inv_i,
        "c_p0": c_p0,
        "c_pi": c_pi,
        "c_inv": c_inv,
        "c_default": c_default,
        "investment_margin_i": (pi_old - p0_old).abs(),
        "investment_switch_i": old_i != new_i,
        "default_flip": survive_old != survive_new,
    }


def projection_movement_decomposition(
    p_prev: torch.Tensor,
    p_current: torch.Tensor,
    y_prev: torch.Tensor,
    y_current: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Index-correct identity for transition k -> k+1, k>=1."""
    delta_p = p_current - p_prev
    delta_target_prev = y_current - y_prev
    e_current = p_current - y_current
    e_prev = p_prev - y_prev
    delta_e = e_current - e_prev
    return {
        "delta_p": delta_p,
        "delta_target_prev": delta_target_prev,
        "e_current": e_current,
        "e_prev": e_prev,
        "delta_e": delta_e,
        "closure": delta_p - delta_target_prev - delta_e,
    }


@dataclass
class BucketAccumulator:
    count: float = 0.0
    weight: float = 0.0
    abs_component: float = 0.0
    flips: float = 0.0

    def add(self, mask: torch.Tensor, weights: torch.Tensor, component: torch.Tensor, flips: torch.Tensor) -> None:
        use = mask & torch.isfinite(weights) & torch.isfinite(component)
        if not bool(use.any()):
            return
        self.count += float(use.sum())
        self.weight += float(weights[use].sum())
        self.abs_component += float((weights[use] * component[use].abs()).sum())
        self.flips += float((weights[use] * flips[use].to(weights.dtype)).sum())


def _bucket_label(lo: float, hi: float) -> str:
    return f"[{lo:g},{hi:g})" if math.isfinite(hi) else f"[{lo:g},inf)"


def _load_setup(run: Path, device: torch.device, limit: Optional[int]):
    provenance = _load_json(run / "metadata" / "provenance.json")
    source = Path(provenance["source_run"])
    source_checkpoint = Path(provenance["source_checkpoint"])
    hp_values = _load_json(source / "metadata" / "hyperparams.json")
    config_values = _load_json(source / "metadata" / "config_snapshot.json")
    economic = AnalysisEconomicConfig.from_dict(config_values)
    for name, value in economic.to_dict().items():
        setattr(Config, name, value)
    Config.DEVICE = device
    hp = _restore_hyperparams(hp_values)
    models, firm_target, loader = load_training_warmstart(
        source_checkpoint,
        device,
        expected_episode=1,
        expected_q_parameterization="direct",
    )
    # Empty by design: the analysis must be structurally incapable of an optimizer step.
    episode = Episode(
        models=models,
        optimizers={},
        config=Config,
        hyperparams=hp,
        device=device,
        episode_id=1,
        firm_target=firm_target,
        q_checkpoint_loaded=True,
    )
    parent_payload = torch.load(run / "fixed_point" / "collocation_val.pt", map_location="cpu", weights_only=False)
    child_all = torch.load(run / "fixed_point" / "child_shock_bank.pt", map_location="cpu", weights_only=False)
    child_payload = child_all["validation"]
    n = int(parent_payload["parent_states"].shape[0])
    if limit is not None:
        n = min(n, int(limit))
    parents = parent_payload["parent_states"][:n].to(device)
    children = child_payload["children"][:n].to(device)
    return episode, hp, models, parents, children, parent_payload, child_payload, provenance, loader


def _checkpoint_paths(run: Path, max_transitions: int) -> list[Path]:
    paths = [run / "checkpoints" / f"sweep_{sweep:03d}.pt" for sweep in range(max_transitions + 1)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing P-only checkpoints: {missing}")
    return paths


def _load_policy_state(model: torch.nn.Module, path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "inner_fixed_point_v1":
        raise ValueError(f"Unexpected checkpoint format: {path}")
    model.load_state_dict(payload["policy_value"], strict=True)
    model.eval().requires_grad_(False)
    return payload


def _expanded_inputs(episode: Episode, parent: torch.Tensor, children: torch.Tensor):
    child_list = list(children.unbind(dim=1))
    lo = float(getattr(episode.hyperparams, "pv_m_clamp_min", 0.7))
    hi = float(getattr(episode.hyperparams, "pv_m_clamp_max", 1.3))
    raw_m, used_m = episode._build_policy_m_lists(parent, child_list, lo, hi)
    child_exp, _, used_exp, weights = episode._expand_policy_expectation_children(
        child_list, raw_m, used_m
    )
    return torch.stack(child_exp, dim=1), torch.stack(used_exp, dim=1), weights


def _teacher(episode: Episode, model: torch.nn.Module) -> BPGridTeacher:
    return BPGridTeacher.from_hyperparams(
        model,
        episode.loss_fns["p0"],
        episode.loss_fns["pi"],
        episode.hyperparams,
        q_target_model=model,
    )


def _eval_production(
    teacher: BPGridTeacher,
    parent: torch.Tensor,
    children: torch.Tensor,
    m_values: torch.Tensor,
    weights: torch.Tensor,
    branch: str,
) -> dict[str, torch.Tensor]:
    return teacher.compute(
        parent,
        children,
        m_values,
        branch=branch,
        child_weights=weights,
    )


def _eval_grid(
    teacher: BPGridTeacher,
    parent: torch.Tensor,
    children: torch.Tensor,
    m_values: torch.Tensor,
    weights: torch.Tensor,
    branch: str,
    bp_grid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # _evaluate_grid is the production objective implementation.  Caller keeps
    # chunks within the configured expanded-state cap.
    return teacher._evaluate_grid(
        parent,
        children,
        m_values,
        bp_grid,
        branch=branch,
        child_weights=weights,
    )


def _i_grid_values(model: torch.nn.Module, child_states: torch.Tensor, chunk: int = 65536):
    flat = child_states.reshape(-1, child_states.shape[-1])[:, :7]
    i_grid = torch.linspace(
        0.0,
        float(model.i_threshold),
        steps=max(int(model.i_grid_size), 2),
        device=flat.device,
        dtype=flat.dtype,
    )
    p0_parts, pi_parts = [], []
    for i_value in i_grid:
        for start in range(0, flat.shape[0], chunk):
            states = flat[start:start + chunk].clone()
            states[:, 3] = i_value
            p0, pi = model._value_outputs(states)
            p0_parts.append(p0.reshape(-1))
            pi_parts.append(pi.reshape(-1))
    per_i = math.ceil(flat.shape[0] / chunk)
    p0_rows, pi_rows = [], []
    for index in range(i_grid.numel()):
        p0_rows.append(torch.cat(p0_parts[index * per_i:(index + 1) * per_i]))
        pi_rows.append(torch.cat(pi_parts[index * per_i:(index + 1) * per_i]))
    shape = (i_grid.numel(),) + tuple(child_states.shape[:2])
    return torch.stack(p0_rows).reshape(shape), torch.stack(pi_rows).reshape(shape)


def _child_channels(
    old_model: torch.nn.Module,
    new_model: torch.nn.Module,
    parent: torch.Tensor,
    children: torch.Tensor,
    bp_old: torch.Tensor,
) -> dict[str, torch.Tensor]:
    child_states, _ = _expand_grid_children(
        children,
        bp_old,
        parent[:, 0:1],
        parent[:, 2:3],
    )
    # One candidate only -> [B,1,J,D].
    child_states = child_states[:, 0]
    p0_old, pi_old = _i_grid_values(old_model, child_states)
    p0_new, pi_new = _i_grid_values(new_model, child_states)
    return child_equity_decomposition(p0_old, pi_old, p0_new, pi_new)


def _weighted_parent(child: torch.Tensor, m_values: torch.Tensor, weights: torch.Tensor, factor: float) -> torch.Tensor:
    use_weights = normalize_child_weights(
        weights,
        n_parent=child.shape[0],
        n_child=child.shape[1],
        device=child.device,
        dtype=child.dtype,
    )
    return factor * (use_weights * m_values.squeeze(-1) * child).sum(dim=1, keepdim=True)


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _line_plot(frame: pd.DataFrame, columns: Sequence[str], path: Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(9, 5))
    for column in columns:
        plt.plot(frame["transition"], frame[column], marker="o", label=column)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    _save_plot(path)


def _build_figures(summary: pd.DataFrame, eta: pd.DataFrame, default: pd.DataFrame,
                   investment: pd.DataFrame, bp_margin: pd.DataFrame, figures: Path) -> list[Path]:
    made: list[Path] = []
    for branch in ("p0", "pi"):
        sub = summary[summary.branch == branch]
        p = figures / f"target_vs_channels_{branch}.png"
        _line_plot(sub, ["delta_target_abs_mean"] + [f"{c}_abs_mean" for c in CHANNELS], p,
                   f"{branch.upper()} target movement and channels", "normalized mean absolute movement")
        made.append(p)
        p = figures / f"directional_attribution_{branch}.png"
        _line_plot(sub, [f"align_{c[2:] if c.startswith('c_') else c}" for c in CHANNELS], p,
                   f"{branch.upper()} directional attribution", "alignment share")
        made.append(p)
    plot_specs = [
        ("continuation_vs_bp.png", ["c_cont_abs_mean", "c_bp_abs_mean"], "Continuation vs BP/search-grid", "normalized abs mean"),
        ("child_channels.png", ["c_p0_abs_mean", "c_pi_abs_mean", "c_inv_abs_mean", "c_default_abs_mean"], "Continuation subchannels", "normalized abs mean"),
        ("empirical_gain.png", ["empirical_gain_l2", "empirical_gain_mean_abs", "empirical_gain_p90_ratio"], "Empirical Bellman movement gain", "ratio"),
        ("target_vs_projection.png", ["delta_target_prev_abs_mean", "projection_error_change_abs_mean", "delta_p_abs_mean"], "Actual P movement: prior target vs projection change", "normalized abs mean"),
        ("closure.png", ["closure_abs_mean", "closure_abs_p90", "closure_abs_max"], "Exact decomposition closure", "raw absolute error"),
    ]
    for name, columns, title, ylabel in plot_specs:
        p = figures / name
        _line_plot(summary, columns, p, title, ylabel)
        made.append(p)
    plt.figure(figsize=(9, 5))
    for (branch, eta_value), sub in eta.groupby(["branch", "eta"]):
        plt.plot(sub.transition, sub.delta_target_abs_mean, marker="o", label=f"{branch} eta={eta_value}")
    plt.title("Target movement by current refinancing state")
    plt.xticks(rotation=45); plt.grid(alpha=.25); plt.legend()
    p = figures / "eta_comparison.png"; _save_plot(p); made.append(p)
    for frame, column, name, title in (
        (default, "abs_component_share", "default_buckets.png", "Default channel by old |Phat|"),
        (investment, "abs_component_share", "investment_buckets.png", "Investment channel by old |PI-P0|"),
    ):
        plt.figure(figsize=(9, 5))
        agg = frame.groupby("bucket", sort=False)[column].mean()
        agg.plot(kind="bar")
        plt.title(title); plt.ylabel(column); plt.xticks(rotation=35)
        p = figures / name; _save_plot(p); made.append(p)
    plt.figure(figsize=(8, 5))
    if not bp_margin.empty:
        agg = bp_margin.groupby("margin_quantile")["c_bp_abs_mean"].mean()
        agg.plot(kind="bar")
    plt.title("BP/search-grid channel by old top-2-margin quantile")
    plt.ylabel("normalized abs mean")
    p = figures / "bp_margin.png"; _save_plot(p); made.append(p)
    return made


def _render_report(
    path: Path,
    run: Path,
    metadata: Mapping[str, Any],
    summary: pd.DataFrame,
    cashflow: pd.DataFrame,
    invariant: Mapping[str, Any],
    outputs: Sequence[Path],
) -> None:
    mean_align = summary.groupby("branch")[[
        "align_p0", "align_pi", "align_inv", "align_default", "align_bp"
    ]].mean()
    channel_mean = summary.groupby("branch")[[f"{c}_abs_mean" for c in CHANNELS]].mean()
    projection = summary.groupby("branch")[["align_target_to_delta_p", "align_projection_to_delta_p"]].mean()
    dominant = {}
    for branch in ("p0", "pi"):
        row = mean_align.loc[branch]
        dominant[branch] = row.abs().idxmax().replace("align_", "")
    lines = [
        "# P-only Bellman movement channel decomposition",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- P-only run: `{run}`",
        f"- Validation parents: {metadata['validation_parents']}",
        f"- Common diagnostic grid: {metadata['common_grid_size']} points",
        f"- Analysis-only optimizer count: {metadata['optimizer_count']} (no step path invoked)",
        "",
        "## Executive result",
        "",
        f"The largest mean directional channel is **{dominant['p0']} for P0** and **{dominant['pi']} for PI**. "
        "These are empirical movement attributions, not a Jacobian or spectral-radius estimate.",
        "",
        "## Code-confirmed operator",
        "",
        "For each branch, the production teacher evaluates cash flow using the frozen direct-Q claim and adds "
        "the hard child-equity continuation. `cal_phats` averages `max(P0,PI)` over the model's exact internal "
        "investment grid; `total_equity(..., hard=True)` is `relu(Phat)`. Exact eta enumeration, normalized "
        "child weights, and the configured M clamp are reused. Candidate Phat does not gate direct-Q issuance.",
        "",
        "## Invariants",
        "",
    ]
    for key, value in invariant.items():
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "Cashflow comparisons:",
        "",
        cashflow.to_markdown(index=False),
        "",
        "## Mean normalized directional attribution",
        "",
        mean_align.to_markdown(),
        "",
        "## Mean normalized absolute channel movement",
        "",
        channel_mean.to_markdown(),
        "",
        "## Target-versus-projection attribution to actual network movement",
        "",
        projection.to_markdown(),
        "",
        "## Interpretation",
        "",
        "The exact closure columns determine whether attribution is admissible. Production-grid `C_bp` is "
        "reported as a BP/search-grid channel because the adaptive local grid need not contain the previous "
        "sweep's optimum. The separate fixed 101-point grid is the economic re-optimization diagnostic and "
        "must be nonnegative up to floating-point noise. Default and investment bucket files localize switching "
        "to their respective boundaries; eta=0 is the natural no-refinancing control.",
        "",
        "`empirical_gain_*` measures movement amplification along this realized trajectory only. It is not "
        "evidence that a spectral radius equals the reported ratio.",
        "",
        "## Outputs",
        "",
    ]
    lines.extend(f"- `{item}`" for item in outputs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run = args.run.expanduser().resolve()
    output = (args.output_dir or run / "analysis" / f"p_only_movement_channels_{date.today():%Y%m%d}").expanduser().resolve()
    report = (args.report or output / "reports" / f"p_only_movement_channel_decomposition_{date.today():%Y%m%d}.md").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_grad_enabled(False)
    episode, hp, models, parents, base_children, parent_payload, child_payload, provenance, loader = _load_setup(
        run, device, args.validation_limit
    )
    if episode.optimizers:
        raise RuntimeError("Analysis must not own optimizers")
    checkpoints = _checkpoint_paths(run, args.max_transitions)
    base_model = models["policy_value"]
    checkpoint_payloads = []
    q_hashes, p_hashes, bp_hashes = [], [], []
    states_cpu: list[dict[str, torch.Tensor]] = []
    for path in checkpoints:
        payload = _load_policy_state(base_model, path, device)
        checkpoint_payloads.append(payload)
        q_hashes.append(named_parameter_hash(base_model, Q_PREFIXES))
        p_hashes.append(named_parameter_hash(base_model, P_PREFIXES))
        bp_hashes.append(named_parameter_hash(base_model, BP_PREFIXES))
        states_cpu.append({name: value.detach().cpu().clone() for name, value in base_model.state_dict().items()})
    if len(set(q_hashes)) != 1:
        raise RuntimeError("Q parameter hash changed across P-only checkpoints")
    if len(set(bp_hashes)) != 1:
        raise RuntimeError("BP-head parameter hash changed across P-only checkpoints")
    expected_parent_hash = parent_payload.get("fingerprint")
    expected_child_hash = child_payload.get("fingerprint")
    parent_hash = _bank_hash_parent(parent_payload)
    child_hash = _bank_hash_children(child_payload)
    if expected_parent_hash != parent_hash or expected_child_hash != child_hash:
        raise RuntimeError("Serialized fixed-bank fingerprint mismatch")
    for payload in checkpoint_payloads:
        hashes = payload.get("bank_hashes", {})
        if hashes.get("validation_parent") != parent_hash or hashes.get("validation_children") != child_hash:
            raise RuntimeError("Checkpoint bank fingerprints do not match loaded validation banks")

    n = parents.shape[0]
    bs = int(args.batch_size)
    production: dict[tuple[int, str], dict[str, torch.Tensor]] = defaultdict(dict)
    function_values: dict[tuple[int, str], torch.Tensor] = {}
    common: dict[tuple[int, str], dict[str, torch.Tensor]] = defaultdict(dict)
    common_grid_values = torch.linspace(0.0, 1.0, args.common_grid_size, device=device, dtype=parents.dtype)

    # Cache every checkpoint's network values and both target surfaces once.
    for sweep, state in enumerate(states_cpu):
        base_model.load_state_dict(state, strict=True)
        base_model.eval().requires_grad_(False)
        teacher = _teacher(episode, base_model)
        function_p0, function_pi, scale_parts = [], [], []
        prod_parts = {branch: defaultdict(list) for branch in ("p0", "pi")}
        common_parts = {branch: defaultdict(list) for branch in ("p0", "pi")}
        for start in range(0, n, bs):
            stop = min(start + bs, n)
            parent = parents[start:stop]
            children, m_values, weights = _expanded_inputs(episode, parent, base_children[start:stop])
            p0, pi = base_model._value_outputs(parent)
            function_p0.append(p0.cpu()); function_pi.append(pi.cpu())
            scale_parts.append(base_model.equity_value_scale(parent).detach().cpu())
            for branch in ("p0", "pi"):
                result = _eval_production(teacher, parent, children, m_values, weights, branch)
                for key in ("value_star", "bp_star", "bp_star_grid", "top2_margin", "fine_top2_margin"):
                    prod_parts[branch][key].append(result[key].cpu())
                # Keep common-grid batches small enough for the hard expanded-state cap.
                common_chunk = max(1, teacher.max_expanded_states // (args.common_grid_size * children.shape[1]))
                for local in range(0, parent.shape[0], common_chunk):
                    local_stop = min(local + common_chunk, parent.shape[0])
                    grid = common_grid_values.reshape(1, -1).expand(local_stop - local, -1)
                    evaluated = _eval_grid(
                        teacher, parent[local:local_stop], children[local:local_stop], m_values[local:local_stop],
                        weights[local:local_stop], branch, grid,
                    )
                    for key in ("value_grid", "cashflow_grid_mean"):
                        common_parts[branch][key].append(evaluated[key].cpu())
        function_values[(sweep, "p0")] = torch.cat(function_p0)
        function_values[(sweep, "pi")] = torch.cat(function_pi)
        function_values[(sweep, "scale")] = torch.cat(scale_parts)
        for branch in ("p0", "pi"):
            production[(sweep, branch)] = {key: torch.cat(value) for key, value in prod_parts[branch].items()}
            common[(sweep, branch)] = {key: torch.cat(value) for key, value in common_parts[branch].items()}

    cashflow_rows = []
    for old, new in ((0, 1), (4, 5), (9, 10)):
        if new > args.max_transitions:
            continue
        for branch in ("p0", "pi"):
            diff = common[(new, branch)]["cashflow_grid_mean"] - common[(old, branch)]["cashflow_grid_mean"]
            cashflow_rows.append({
                "transition": f"{old}->{new}", "branch": branch,
                "mean_abs_cashflow_diff": float(diff.abs().mean()),
                "p90_abs_cashflow_diff": float(torch.quantile(diff.abs(), .90)),
                "max_abs_cashflow_diff": float(diff.abs().max()),
            })
    cashflow_frame = pd.DataFrame(cashflow_rows)
    cashflow_tolerance = 2e-6
    if not cashflow_frame.empty and float(cashflow_frame.max_abs_cashflow_diff.max()) > cashflow_tolerance:
        cashflow_frame.to_csv(output / "cashflow_invariance_failure.csv", index=False)
        raise RuntimeError("P-only cashflow surface is not invariant; decomposition stopped")

    summary_rows, state_rows, eta_rows = [], [], []
    default_acc: dict[tuple, BucketAccumulator] = defaultdict(BucketAccumulator)
    invest_acc: dict[tuple, BucketAccumulator] = defaultdict(BucketAccumulator)
    closure_maxima = defaultdict(float)
    eta0_bp_max = 0.0
    common_bp_min = 0.0

    old_model = deepcopy(base_model).to(device).eval().requires_grad_(False)
    new_model = deepcopy(base_model).to(device).eval().requires_grad_(False)
    for k in range(args.max_transitions):
        old_model.load_state_dict(states_cpu[k], strict=True)
        new_model.load_state_dict(states_cpu[k + 1], strict=True)
        new_teacher = _teacher(episode, new_model)
        for branch in ("p0", "pi"):
            all_parent = defaultdict(list)
            for start in range(0, n, bs):
                stop = min(start + bs, n)
                parent = parents[start:stop]
                children, m_values, weights = _expanded_inputs(episode, parent, base_children[start:stop])
                bp_old = production[(k, branch)]["bp_star"][start:stop].to(device)
                forced_new = _eval_grid(new_teacher, parent, children, m_values, weights, branch, bp_old)
                y_old = production[(k, branch)]["value_star"][start:stop].to(device)
                y_new = production[(k + 1, branch)]["value_star"][start:stop].to(device)
                y_new_old_b = forced_new["value_grid"][:, :1]
                c_cont = y_new_old_b - y_old
                c_bp = y_new - y_new_old_b
                child = _child_channels(old_model, new_model, parent, children, bp_old)
                factor = 1.0 if branch == "p0" else float(Config.G)
                c_p0 = _weighted_parent(child["c_p0"], m_values, weights, factor)
                c_pi = _weighted_parent(child["c_pi"], m_values, weights, factor)
                c_inv = _weighted_parent(child["c_inv"], m_values, weights, factor)
                c_default = _weighted_parent(child["c_default"], m_values, weights, factor)
                delta_y = y_new - y_old
                continuation_closure = c_cont - c_p0 - c_pi - c_inv - c_default
                full_closure = delta_y - c_p0 - c_pi - c_inv - c_default - c_bp
                child_phat_closure = child["phat_new"] - child["phat_old"] - child["c_p0_raw"] - child["c_pi_raw"] - child["c_inv_raw"]
                hard_closure = child["p_new"] - child["p_old"] - child["c_p0"] - child["c_pi"] - child["c_inv"] - child["c_default"]
                closure_maxima["child_phat"] = max(closure_maxima["child_phat"], float(child_phat_closure.abs().max()))
                closure_maxima["hard_child"] = max(closure_maxima["hard_child"], float(hard_closure.abs().max()))
                closure_maxima["continuation"] = max(closure_maxima["continuation"], float(continuation_closure.abs().max()))
                closure_maxima["full_target"] = max(closure_maxima["full_target"], float(full_closure.abs().max()))
                eta0 = parent[:, 2:3] <= .5
                if bool(eta0.any()):
                    eta0_bp_max = max(eta0_bp_max, float(c_bp[eta0].abs().max()))
                for key, value in {
                    "delta_target": delta_y, "c_cont": c_cont, "c_p0": c_p0, "c_pi": c_pi,
                    "c_inv": c_inv, "c_default": c_default, "c_bp": c_bp,
                    "closure": full_closure,
                }.items():
                    all_parent[key].append(value.cpu())
                all_parent["scale"].append(function_values[(k, "scale")][start:stop])
                all_parent["eta"].append(parent[:, 2:3].cpu())
                all_parent["bp_old"].append(bp_old.cpu())
                all_parent["bp_new"].append(production[(k + 1, branch)]["bp_star"][start:stop])
                all_parent["top2_margin"].append(production[(k, branch)]["top2_margin"][start:stop])
                all_parent["fine_top2_margin"].append(production[(k, branch)]["fine_top2_margin"][start:stop])
                all_parent["investment_switch_share"].append(child["investment_switch_i"].float().mean(dim=(0, 2), keepdim=False).reshape(-1, 1).cpu())
                all_parent["default_flip_share"].append(child["default_flip"].float().mean(dim=1, keepdim=True).cpu())

                norm_weights = normalize_child_weights(weights, n_parent=parent.shape[0], n_child=children.shape[1], device=device, dtype=parent.dtype)
                child_weight = factor * norm_weights * m_values.squeeze(-1)
                for lo, hi in zip(FIXED_DEFAULT_BUCKETS[:-1], FIXED_DEFAULT_BUCKETS[1:]):
                    margin = child["phat_old"].abs()
                    mask = (margin >= lo) & (margin < hi)
                    default_acc[(k, branch, lo, hi)].add(mask, child_weight, child["c_default"], child["default_flip"])
                i_count = child["investment_margin_i"].shape[0]
                i_weight = child_weight.unsqueeze(0).expand(i_count, -1, -1) / float(i_count)
                i_component = child["c_inv_i"] * (child["phat_old"] > 0).to(parent.dtype).unsqueeze(0)
                for lo, hi in zip(FIXED_INVEST_BUCKETS[:-1], FIXED_INVEST_BUCKETS[1:]):
                    margin = child["investment_margin_i"]
                    mask = (margin >= lo) & (margin < hi)
                    invest_acc[(k, branch, lo, hi)].add(mask, i_weight, i_component, child["investment_switch_i"])

            combined = {key: torch.cat(value) for key, value in all_parent.items()}
            scale = combined["scale"].clamp_min(1e-8)
            transition = f"{k}->{k+1}"
            row: dict[str, Any] = {"transition": transition, "sweep_from": k, "sweep_to": k + 1, "branch": branch}
            for key in ("delta_target", "c_cont", *CHANNELS, "closure"):
                row.update(_stats(combined[key] / scale if key != "closure" else combined[key], key))
            for channel in CHANNELS:
                label = channel[2:] if channel.startswith("c_") else channel
                row[f"align_{label}"] = _alignment(combined[channel], combined["delta_target"], scale)
            row["alignment_sum"] = sum(row[f"align_{c[2:]}"] for c in CHANNELS)
            delta_p = function_values[(k + 1, branch)] - function_values[(k, branch)]
            row["empirical_gain_l2"] = _ratio(combined["delta_target"] / scale, delta_p / scale, "l2")
            row["empirical_gain_mean_abs"] = _ratio(combined["delta_target"] / scale, delta_p / scale, "mean_abs")
            row["empirical_gain_p90_ratio"] = _ratio(combined["delta_target"] / scale, delta_p / scale, "p90")
            fit_error = function_values[(k + 1, branch)] - production[(k, branch)]["value_star"]
            row.update(_stats(fit_error / scale, "fit_error"))
            row["fit_error_mean"] = row["fit_error_abs_mean"]
            row["fit_error_p90"] = row["fit_error_abs_p90"]
            if k >= 1:
                projection = projection_movement_decomposition(
                    function_values[(k, branch)], function_values[(k + 1, branch)],
                    production[(k - 1, branch)]["value_star"], production[(k, branch)]["value_star"],
                )
                proj_closure = projection["closure"]
                closure_maxima["projection"] = max(closure_maxima["projection"], float(proj_closure.abs().max()))
                row.update(_stats(projection["delta_p"] / scale, "delta_p"))
                row.update(_stats(projection["delta_target_prev"] / scale, "delta_target_prev"))
                row.update(_stats(projection["delta_e"] / scale, "projection_error_change"))
                row["projection_error_change_mean"] = row["projection_error_change_abs_mean"]
                row["projection_error_change_p90"] = row["projection_error_change_abs_p90"]
                row["align_target_to_delta_p"] = _alignment(projection["delta_target_prev"], projection["delta_p"], scale)
                row["align_projection_to_delta_p"] = _alignment(projection["delta_e"], projection["delta_p"], scale)
                row["projection_alignment_sum"] = row["align_target_to_delta_p"] + row["align_projection_to_delta_p"]
            else:
                for key in ("delta_p", "delta_target_prev", "projection_error_change"):
                    row.update({name: float("nan") for name in _stats(torch.empty(0), key)})
                row.update({"projection_error_change_mean": float("nan"), "projection_error_change_p90": float("nan")})
                row.update({"align_target_to_delta_p": float("nan"), "align_projection_to_delta_p": float("nan"), "projection_alignment_sum": float("nan")})

            old_grid = common[(k, branch)]["value_grid"]
            new_grid = common[(k + 1, branch)]["value_grid"]
            old_idx = old_grid.argmax(dim=1, keepdim=True)
            new_idx = new_grid.argmax(dim=1, keepdim=True)
            new_at_old = torch.gather(new_grid, 1, old_idx)
            new_max = torch.gather(new_grid, 1, new_idx)
            c_bp_common = new_max - new_at_old
            common_bp_min = min(common_bp_min, float(c_bp_common.min()))
            row.update(_stats(c_bp_common / scale, "c_bp_common"))
            row["negative_c_bp_share"] = float((combined["c_bp"] < -1e-7).float().mean())
            row["c_bp_min"] = float(combined["c_bp"].min())
            row["c_bp_p01"] = float(torch.quantile(combined["c_bp"], .01))
            row["negative_c_bp_common_share"] = float((c_bp_common < -1e-7).float().mean())
            row["bp_star_abs_change_mean"] = float((combined["bp_new"] - combined["bp_old"]).abs().mean())
            row["bp_star_abs_change_p90"] = float(torch.quantile((combined["bp_new"] - combined["bp_old"]).abs(), .90))
            row["bp_star_flip_share"] = float(((combined["bp_new"] - combined["bp_old"]).abs() > 1e-6).float().mean())
            summary_rows.append(row)

            for eta_value in (0, 1):
                mask = combined["eta"].reshape(-1) == eta_value
                erow = {"transition": transition, "branch": branch, "eta": eta_value, "count": int(mask.sum())}
                for key in ("delta_target", "c_cont", *CHANNELS):
                    erow.update(_stats((combined[key][mask] / scale[mask]), key))
                eta_rows.append(erow)

            for index in range(n):
                state_rows.append({
                    "parent_id": index, "transition": transition, "branch": branch,
                    "eta": int(combined["eta"][index]), "scale": float(scale[index]),
                    **{key: float(combined[key][index]) for key in ("delta_target", "c_cont", *CHANNELS, "closure")},
                    "bp_star_old": float(combined["bp_old"][index]),
                    "bp_star_new": float(combined["bp_new"][index]),
                    "top2_margin": float(combined["top2_margin"][index]),
                    "fine_top2_margin": float(combined["fine_top2_margin"][index]),
                    "investment_switch_share": float(combined["investment_switch_share"][index]),
                    "default_flip_share": float(combined["default_flip_share"][index]),
                })

    summary = pd.DataFrame(summary_rows)
    eta_frame = pd.DataFrame(eta_rows)
    state_frame = pd.DataFrame(state_rows)
    default_rows, investment_rows = [], []
    for (k, branch, lo, hi), acc in default_acc.items():
        total = sum(v.abs_component for key, v in default_acc.items() if key[0] == k and key[1] == branch)
        state_subset = state_frame[(state_frame.transition == f"{k}->{k+1}") & (state_frame.branch == branch)]
        continuation_abs_sum = float(state_subset.c_cont.abs().sum())
        default_rows.append({
            "transition": f"{k}->{k+1}", "branch": branch, "bucket_scheme": "fixed", "bucket": _bucket_label(lo, hi),
            "child_count": int(acc.count), "weighted_mass": acc.weight,
            "weighted_abs_component": acc.abs_component,
            "abs_component_share": acc.abs_component / total if total > 0 else float("nan"),
            "abs_component_over_total_parent_abs_continuation": acc.abs_component / max(continuation_abs_sum, 1e-20),
            "default_flip_share": acc.flips / acc.weight if acc.weight > 0 else float("nan"),
        })
    for (k, branch, lo, hi), acc in invest_acc.items():
        total = sum(v.abs_component for key, v in invest_acc.items() if key[0] == k and key[1] == branch)
        investment_rows.append({
            "transition": f"{k}->{k+1}", "branch": branch, "bucket_scheme": "fixed", "bucket": _bucket_label(lo, hi),
            "node_count": int(acc.count), "weighted_mass": acc.weight,
            "weighted_abs_component": acc.abs_component,
            "abs_component_share": acc.abs_component / total if total > 0 else float("nan"),
            "investment_switch_share": acc.flips / acc.weight if acc.weight > 0 else float("nan"),
        })
    default_frame = pd.DataFrame(default_rows)
    investment_frame = pd.DataFrame(investment_rows)
    bp_rows = []
    state_frame["normalized_c_bp_abs"] = state_frame.c_bp.abs() / state_frame.scale.clip(lower=1e-8)
    for (transition, branch), sub in state_frame.groupby(["transition", "branch"]):
        ranked = pd.qcut(sub.top2_margin.rank(method="first"), 4, labels=["Q1 weakest", "Q2", "Q3", "Q4 strongest"])
        for label, group in sub.groupby(ranked, observed=True):
            bp_rows.append({
                "transition": transition, "branch": branch, "margin_quantile": str(label), "count": len(group),
                "top2_margin_mean": group.top2_margin.mean(), "c_bp_abs_mean": group.normalized_c_bp_abs.mean(),
                "bp_star_abs_change_mean": (group.bp_star_new - group.bp_star_old).abs().mean(),
            })
    bp_frame = pd.DataFrame(bp_rows)

    outputs = [
        output / "p_movement_channel_summary.csv",
        output / "p_movement_channel_by_eta.csv",
        output / "p_movement_channel_by_default_margin.csv",
        output / "p_movement_channel_by_investment_margin.csv",
        output / "p_movement_channel_by_bp_margin.csv",
        output / "p_movement_channel_statewise.parquet",
        output / "cashflow_invariance.csv",
    ]
    for frame, path in zip(
        (summary, eta_frame, default_frame, investment_frame, bp_frame), outputs[:5]
    ):
        frame.to_csv(path, index=False)
    cashflow_frame.to_csv(outputs[6], index=False)
    try:
        state_frame.to_parquet(outputs[5], index=False)
    except (ImportError, ModuleNotFoundError):
        outputs[5] = output / "p_movement_channel_statewise.csv.gz"
        state_frame.to_csv(outputs[5], index=False, compression="gzip")

    figures = _build_figures(summary, eta_frame, default_frame, investment_frame, bp_frame, output / "figures")
    outputs.extend(figures)
    invariant = {
        "cashflow_surface_fixed": bool(cashflow_frame.max_abs_cashflow_diff.max() <= cashflow_tolerance),
        "child_phat_closure_max": closure_maxima["child_phat"],
        "hard_child_equity_closure_max": closure_maxima["hard_child"],
        "parent_continuation_closure_max": closure_maxima["continuation"],
        "full_target_closure_max": closure_maxima["full_target"],
        "projection_movement_closure_max": closure_maxima["projection"],
        "eta0_c_bp_abs_max": eta0_bp_max,
        "common_grid_c_bp_min": common_bp_min,
        "validation_parent_fingerprint": parent_hash,
        "validation_child_fingerprint": child_hash,
        "q_hash_identical": len(set(q_hashes)) == 1,
        "q_hash": q_hashes[0],
        "bp_hash_identical": len(set(bp_hashes)) == 1,
        "optimizer_count": len(episode.optimizers),
        "optimizer_or_training_steps": 0,
    }
    if max(closure_maxima.values()) > args.closure_atol:
        raise RuntimeError(f"Exact decomposition closure exceeded tolerance: {dict(closure_maxima)}")
    if eta0_bp_max > args.closure_atol:
        raise RuntimeError(f"eta=0 BP channel is nonzero: {eta0_bp_max}")
    if common_bp_min < -args.closure_atol:
        raise RuntimeError(f"Common-grid reoptimization channel is negative: {common_bp_min}")
    metadata = {
        "run": str(run), "device": str(device), "validation_parents": n,
        "common_grid_size": args.common_grid_size, "optimizer_count": len(episode.optimizers),
        "checkpoints": [str(path) for path in checkpoints], "q_hashes": q_hashes,
        "p_hashes": p_hashes, "bp_hashes": bp_hashes,
        "provenance": provenance, "loader_provenance": loader, "invariants": invariant,
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    outputs.append(output / "analysis_metadata.json")
    _render_report(report, run, metadata, summary, cashflow_frame, invariant, outputs)
    outputs.append(report)
    print(json.dumps({"status": "complete", "output_dir": str(output), "report": str(report), "invariants": invariant}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
