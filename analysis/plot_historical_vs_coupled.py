#!/usr/bin/env python3
"""Plot historical P-only Picard diagnostics against coupled block results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--coupled", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    historical = pd.read_csv(args.historical)
    coupled = pd.read_csv(args.coupled)
    historical = historical[historical["mode"] == "p_only"].copy()
    historical["p_residual"] = 0.5 * (
        historical["p0_residual_mean"] + historical["pi_residual_mean"]
    )
    historical["p_movement"] = 0.5 * (
        historical["p0_function_change_mean"]
        + historical["pi_function_change_mean"]
    )
    coupled["p_movement"] = 0.5 * (
        coupled["p0_function_change_mean"]
        + coupled["pi_function_change_mean"]
    )
    comparison = pd.concat([
        pd.DataFrame({
            "algorithm": "historical P-only Picard (Q frozen)",
            "iteration": historical["sweep"],
            "p_residual": historical["p_residual"],
            "q_residual": historical["q_residual_mean"],
            "p_movement": historical["p_movement"],
            "q_movement": historical["q_function_change_mean"],
            "p_p90_magnitude": np.nan,
            "q_p90_magnitude": np.nan,
        }),
        pd.DataFrame({
            "algorithm": "fixed-macro P-Q block solver",
            "iteration": coupled["outer"],
            "p_residual": coupled["p_residual"],
            "q_residual": coupled["q_residual"],
            "p_movement": coupled["p_movement"],
            "q_movement": coupled["q_movement_mean"],
            "p_p90_magnitude": coupled["total_p_p90"],
            "q_p90_magnitude": coupled["q_p90"],
        }),
    ], ignore_index=True)
    comparison.to_csv(output / "historical_picard_vs_coupled.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = (
        ("p_residual", "P residual"),
        ("q_residual", "Q residual"),
        ("p_movement", "P movement"),
        ("q_movement", "Q movement"),
        ("p_p90_magnitude", "P p90 magnitude"),
        ("q_p90_magnitude", "Q p90 magnitude"),
    )
    for axis, (column, title) in zip(axes.reshape(-1), panels):
        for label, frame in comparison.groupby("algorithm"):
            valid = frame[column].notna()
            if valid.any():
                axis.plot(
                    frame.loc[valid, "iteration"], frame.loc[valid, column],
                    marker="o", label=label,
                )
        axis.set(title=title, xlabel="sweep or coupled outer")
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Algorithm comparison (iteration axes are not economically equivalent; "
        "historical run froze Q and omitted magnitude fields)"
    )
    fig.tight_layout()
    fig.savefig(output / "historical_picard_vs_coupled.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
