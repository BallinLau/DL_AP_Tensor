"""Plots and report for the frozen-Macro-Law firm-side fixed point."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_positive(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return np.where(np.isfinite(array) & (array > 0), array, np.nan)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _line_plot(path: Path, sweeps: list[int], series: Mapping[str, Iterable[float]], title: str,
               ylabel: str, *, log: bool = False, thresholds: Iterable[float] = ()) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, values in series.items():
        array = _finite_positive(values) if log else np.asarray(list(values), dtype=float)
        ax.plot(sweeps, array, marker="o", ms=3, label=label)
    for threshold in thresholds:
        if threshold is not None and math.isfinite(float(threshold)) and float(threshold) > 0:
            ax.axhline(float(threshold), ls="--", lw=1, color="black", alpha=0.6)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("inner sweep")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, path)


def _selected_sweeps(available: list[int]) -> list[int]:
    wanted = [0, 1, 2, 5, 10, max(available)]
    return sorted(set(value for value in wanted if value in available))


def _coverage_plots(run_root: Path, plot_dir: Path) -> None:
    train = torch.load(run_root / "fixed_point" / "collocation_train.pt", weights_only=False)
    val = torch.load(run_root / "fixed_point" / "collocation_val.pt", weights_only=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for payload, marker, label in ((train, ".", "train"), (val, "x", "validation")):
        state = payload["parent_states"].numpy()
        kind = payload["source_kind"].numpy()
        for source, color, suffix in ((0, "tab:blue", "on-support"), (1, "tab:orange", "coverage")):
            mask = kind == source
            axes[0, 0].scatter(state[mask, 0], state[mask, 1], s=5, marker=marker,
                               alpha=0.25, color=color, label=f"{label} {suffix}")
            axes[0, 1].scatter(state[mask, 4], state[mask, 5], s=5, marker=marker,
                               alpha=0.25, color=color)
            axes[1, 0].scatter(state[mask, 4], state[mask, 6], s=5, marker=marker,
                               alpha=0.25, color=color)
            axes[1, 1].scatter(state[mask, 5], state[mask, 6], s=5, marker=marker,
                               alpha=0.25, color=color)
    for ax, xlabel, ylabel in zip(
        axes.flat,
        ("b", "x", "x", "Hatcf"),
        ("z", "Hatcf", "LnKF", "LnKF"),
    ):
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Immutable train/validation collocation coverage")
    _save(fig, plot_dir / "collocation_coverage.png")


def _boundary_plots(run_root: Path, plot_dir: Path) -> None:
    grid = torch.load(run_root / "fixed_point" / "macro_anchor_bank.pt", weights_only=False)
    b = grid["b_values"].numpy()
    z = grid["z_values"].numpy()
    boundary_paths = sorted((run_root / "fixed_point" / "boundaries").glob("sweep_*.pt"))
    available = [int(path.stem.split("_")[-1]) for path in boundary_paths]
    selected = _selected_sweeps(available)
    payloads = {
        sweep: torch.load(
            run_root / "fixed_point" / "boundaries" / f"sweep_{sweep:03d}.pt",
            weights_only=False,
        )
        for sweep in selected
    }
    for anchor in range(len(grid["anchor_quantiles"])):
        fig, axes = plt.subplots(2, len(selected), figsize=(3.2 * len(selected), 6), squeeze=False)
        common = [payloads[s]["phat"][anchor].numpy() for s in selected]
        vmax = max(float(np.nanpercentile(np.abs(value), 98)) for value in common)
        vmax = max(vmax, 1e-8)
        for col, sweep in enumerate(selected):
            data = payloads[sweep]
            for eta in (0, 1):
                ax = axes[eta, col]
                image = ax.imshow(data["phat"][anchor, eta].numpy().T, origin="lower",
                                  aspect="auto", extent=(b.min(), b.max(), z.min(), z.max()),
                                  cmap="coolwarm", vmin=-vmax, vmax=vmax)
                ax.contour(b, z, data["default"][anchor, eta].numpy().T.astype(float),
                           levels=[0.5], colors="black", linewidths=0.8)
                ax.set_title(f"sweep {sweep}, eta={eta}")
                ax.set_xlabel("b")
                if col == 0:
                    ax.set_ylabel("z")
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="Phat")
        fig.suptitle(f"Default region evolution: macro anchor {anchor}")
        _save(fig, plot_dir / f"default_heatmaps_anchor_{anchor}.png")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
        for eta in (0, 1):
            for sweep in selected:
                axes[eta].plot(b, payloads[sweep]["cutoff"][anchor, eta].numpy(),
                               label=f"sweep {sweep}", alpha=0.85)
            axes[eta].set(xlabel="b", title=f"eta={eta}")
            axes[eta].grid(alpha=0.2)
        axes[0].set_ylabel("default cutoff zbar(b)")
        axes[1].legend(fontsize=7)
        fig.suptitle(f"Default boundary overlay: macro anchor {anchor}")
        _save(fig, plot_dir / f"default_boundary_overlay_anchor_{anchor}.png")

    focus_anchor = len(grid["anchor_quantiles"]) // 2
    focus_eta = 1
    first, mid, final = selected[0], selected[len(selected) // 2], selected[-1]
    for field in ("q", "p0", "pi", "phat"):
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        for ax, sweep in zip(axes[:3], (first, mid, final)):
            array = payloads[sweep][field][focus_anchor, focus_eta].numpy().T
            im = ax.imshow(array, origin="lower", aspect="auto",
                           extent=(b.min(), b.max(), z.min(), z.max()), cmap="viridis")
            ax.set_title(f"{field}, sweep {sweep}")
            ax.set(xlabel="b", ylabel="z")
            fig.colorbar(im, ax=ax, shrink=0.75)
        before = payloads[selected[-2]][field][focus_anchor, focus_eta].numpy().T if len(selected) > 1 else 0
        diff = payloads[final][field][focus_anchor, focus_eta].numpy().T - before
        lim = max(float(np.nanmax(np.abs(diff))), 1e-10)
        im = axes[3].imshow(diff, origin="lower", aspect="auto",
                            extent=(b.min(), b.max(), z.min(), z.max()), cmap="coolwarm",
                            vmin=-lim, vmax=lim)
        axes[3].set_title(f"final - previous selected")
        axes[3].set(xlabel="b", ylabel="z")
        fig.colorbar(im, ax=axes[3], shrink=0.75)
        fig.suptitle(f"{field} surface evolution (median macro anchor, eta=1)")
        _save(fig, plot_dir / f"surface_{field}_evolution.png")


def _implicit_policy_plot(run_root: Path, plot_dir: Path, sweeps: list[int]) -> None:
    val = torch.load(run_root / "fixed_point" / "collocation_val.pt", weights_only=False)
    states = val["parent_states"].numpy()
    selected = _selected_sweeps([0] + sweeps)
    final_payload = torch.load(
        run_root / "fixed_point" / "surfaces" / f"sweep_{selected[-1]:03d}.pt",
        weights_only=False,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key in zip(axes, ("b0_star", "bi_star")):
        color = final_payload["policies"][key].reshape(-1).numpy()
        sc = ax.scatter(states[:, 0], states[:, 1], c=color, s=8, alpha=0.5,
                        cmap="viridis", vmin=0, vmax=1)
        ax.set(xlabel="b", ylabel="z", title=f"final implicit {key}")
        fig.colorbar(sc, ax=ax)
    _save(fig, plot_dir / "implicit_policy_final.png")


def build_inner_fixed_point_report(run_root: str | Path) -> Path:
    run_root = Path(run_root)
    fixed = run_root / "fixed_point"
    plot_dir = fixed / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = _load_json(fixed / "metrics.json").get("sweeps", [])
    summary = _load_json(fixed / "final_summary.json")
    sweeps = [int(row["sweep"]) for row in metrics]

    for normalized in (False, True):
        key = "normalized" if normalized else "raw"
        for stat in ("mean", "p90"):
            _line_plot(
                plot_dir / f"bellman_{key}_{stat}.png", sweeps,
                {eq.upper(): [row["operator"]["equations"][eq][key][stat] for row in metrics]
                 for eq in ("p0", "pi", "q")},
                f"Bellman residual ({key}, {stat})", f"|R| {stat}", log=True,
            )
    thresholds = summary.get("final_operator", {}).get("bellman_thresholds", {})
    _line_plot(
        plot_dir / "empirical_fixed_point_distance.png", sweeps,
        {
            "normalized mean": [row["operator"]["empirical_fixed_point_distance_mean"] for row in metrics],
            "normalized p90": [row["operator"]["empirical_fixed_point_distance_p90"] for row in metrics],
        },
        "Empirical distance to fixed point", "normalized operator residual", log=True,
        thresholds=(thresholds.get("mean"), thresholds.get("p90")),
    )
    for stat in ("mean", "p90", "max"):
        _line_plot(
            plot_dir / f"successive_function_change_{stat}.png", sweeps,
            {eq.upper(): [row["function_change"][eq][stat] for row in metrics]
             for eq in ("p0", "pi", "q")},
            f"Successive function change ({stat})", f"|F(k)-F(k-1)| {stat}", log=True,
        )
    _line_plot(
        plot_dir / "gomes_q_sup_proxy.png", sweeps,
        {"Delta Q sup proxy": [row["function_change"]["q"]["max"] for row in metrics]},
        "Gomes-style successive Q convergence", "max validation |Q(k)-Q(k-1)|", log=True,
    )
    _line_plot(
        plot_dir / "implicit_policy_change.png", sweeps,
        {
            "b0* p90": [row["implicit_policy"]["b0_star"]["p90"] for row in metrics],
            "bI* p90": [row["implicit_policy"]["bi_star"]["p90"] for row in metrics],
        },
        "Implicit BPGridTeacher policy stability", "p90 |b*(k)-b*(k-1)|", log=True,
        thresholds=(summary["thresholds"].get("bstar_p90_tol"),),
    )
    _line_plot(
        plot_dir / "default_stability.png", sweeps,
        {
            "default area": [row["default_boundary"]["default_area_share"] for row in metrics],
            "default flip": [row["default_boundary"]["default_flip_share"] for row in metrics],
            "cutoff drift p90": [row["default_boundary"]["default_cutoff_drift_p90"] for row in metrics],
        },
        "Default region and boundary stability", "share / z drift", log=False,
        thresholds=(summary["thresholds"].get("default_flip_tol"),),
    )
    _coverage_plots(run_root, plot_dir)
    _boundary_plots(run_root, plot_dir)
    _implicit_policy_plot(run_root, plot_dir, sweeps)

    final_op = summary.get("final_operator") or {}
    equations = final_op.get("equations", {})
    final_change = summary.get("final_function_change") or {}
    final_boundary = summary.get("final_default_boundary") or {}
    final_policy = summary.get("final_implicit_policy") or {}
    report = f"""# Firm-side inner fixed-point report

1. **Numerical inner fixed point found:** `{summary.get('inner_fp_passed')}`
2. **First passing sweep:** `{summary.get('first_pass_sweep')}`
3. **Convergence statement:** {summary.get('statement')}
4. **Final empirical distance:** mean `{summary.get('final_empirical_fp_distance_mean')}`, p90 `{summary.get('final_empirical_fp_distance_p90')}` (normalized fixed-bank operator residual).
5. **Final stability:** ΔP0 p90 `{final_change.get('p0', {}).get('p90')}`, ΔPI p90 `{final_change.get('pi', {}).get('p90')}`, ΔQ p90 `{final_change.get('q', {}).get('p90')}`; default flip `{final_boundary.get('default_flip_share')}`; cutoff drift p90 `{final_boundary.get('default_cutoff_drift_p90')}`; b0* drift p90 `{final_policy.get('b0_star', {}).get('p90')}`; bI* drift p90 `{final_policy.get('bi_star', {}).get('p90')}`.

The result is conditional on the frozen Episode-1 Macro Law/SDF, this immutable parent bank, and this immutable one-step shock bank. It does not establish existence or uniqueness in the continuous state space. `failure_reason={summary.get('failure_reason')}` and `numerical_failure={summary.get('numerical_failure')}`.

## Final Bellman residuals

| equation | raw mean | raw p90 | raw p99 | raw max | normalized mean | normalized p90 | normalized p99 | normalized max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for eq in ("p0", "pi", "q"):
        raw = equations.get(eq, {}).get("raw", {})
        norm = equations.get(eq, {}).get("normalized", {})
        report += "| " + eq.upper() + " | " + " | ".join(
            str(raw.get(key)) for key in ("mean", "p90", "p99", "max")
        ) + " | " + " | ".join(
            str(norm.get(key)) for key in ("mean", "p90", "p99", "max")
        ) + " |\n"
    report += """

## Interpretation

This experiment isolates the firm block of the Gomes-style loop: hold the aggregate forecasting rule fixed, iterate the P0/PI/Q functional equations, and test firm-side convergence. It does not perform the subsequent simulation or aggregate-law update. The default boundary is shown separately because small equity-value movements near zero can discretely change default classification and therefore debt value Q. The `gomes_q_sup_proxy.png` plot reports `max_validation |Q_k-Q_{k-1}|`; the classical tabular tolerances are a reference only and are not used as this neural solver's acceptance rule.

The Bellman plots are fresh current-operator checks: sweep k places F_k on both the left and right sides. The function-change plots measure successive iterates and are not called distance to the fixed point. Boundary plots mark missing zero crossings as gaps. Implicit b* plots come from BPGridTeacher grid search; no BP head is trained.

## Core artifacts

- `metrics.json` / `metrics.csv`: every sweep, gates, hashes, optimizer counts, and timing.
- `final_summary.json`: final verdict, distances, thresholds, and source provenance.
- `collocation_train.pt`, `collocation_val.pt`, `child_shock_bank.pt`: immutable banks and fingerprints.
- `boundaries/`: Phat/default/cutoff and P/Q surfaces on fixed diagnostic grids.
- `surfaces/`: validation-bank function values and implicit b* policies.
- `plots/`: residuals, empirical distance, successive changes, default evolution, P/Q surfaces, policies, and coverage.
"""
    path = fixed / "REPORT.md"
    path.write_text(report, encoding="utf-8")
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    print(build_inner_fixed_point_report(parser.parse_args().run_root))
