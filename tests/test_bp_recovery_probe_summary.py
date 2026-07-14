from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.summarize_bp_recovery_probes import (
    PROBES,
    decision_table,
    load_all_runs,
    pairwise_comparisons,
    expected_summary_path,
)


def make_summary_rows(episode: int, seed: int) -> pd.DataFrame:
    rows = []
    values = {
        "baseline_output_loss": 0.40 + seed * 0.0,
        "branch_only_output_loss": 0.35,
        "bias_recenter_output_loss": 0.25,
        "original_init_logit_loss": 0.20,
    }
    for probe in PROBES:
        rows.append(
            {
                "episode": episode,
                "seed": seed,
                "probe": probe,
                "final_holdout_mae": values[probe],
                "final_holdout_mae_p90": values[probe] + 0.01,
                "final_weighted_holdout_mae": values[probe] + 0.02,
                "final_holdout_pred_target_corr": 0.5,
                "final_holdout_pred_target_spearman": 0.4,
                "final_holdout_bp_pred_std": 0.2,
                "holdout_constant_policy_flag": False,
                "initial_bp_head_grad_norm": 1.0,
                "final_bp_head_grad_norm": 0.5,
                "final_train_pred_target_corr": 0.9,
            }
        )
    return pd.DataFrame(rows)


def write_run(root: Path, episode: int, seed: int) -> None:
    out = root / f"ep{episode}" / f"seed{seed}"
    out.mkdir(parents=True)
    make_summary_rows(episode, seed).to_csv(out / f"ep{episode}_bp_recovery_probe_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "episode": episode,
                "seed": seed,
                "probe": "baseline_output_loss",
                "step": 0,
                "split": "train",
            }
        ]
    ).to_csv(out / f"ep{episode}_bp_recovery_probe_history.csv", index=False)


def test_summary_requires_all_episode_seed_outputs(tmp_path: Path):
    write_run(tmp_path, 1, 12345)
    try:
        load_all_runs(tmp_path, [1], [12345, 23456])
    except FileNotFoundError as exc:
        assert "seed=23456" in str(exc)
    else:
        raise AssertionError("missing seed should fail")


def test_pairwise_comparisons_use_required_probe_pairs(tmp_path: Path):
    for episode in [1, 2]:
        for seed in [12345, 23456, 34567]:
            write_run(tmp_path, episode, seed)
    all_runs = load_all_runs(tmp_path, [1, 2], [12345, 23456, 34567])
    assert set(all_runs["probe"].unique()) == set(PROBES)

    comparisons = pairwise_comparisons(all_runs)
    h002 = comparisons.loc[comparisons["comparison"] == "H002_bias_recenter_minus_baseline"]
    h003 = comparisons.loc[comparisons["comparison"] == "H003_logit_minus_branch_output"]
    mix = comparisons.loc[comparisons["comparison"] == "mix_branch_minus_full_output"]
    assert h002["left_probe"].eq("bias_recenter_output_loss").all()
    assert h002["right_probe"].eq("baseline_output_loss").all()
    assert h003["left_probe"].eq("original_init_logit_loss").all()
    assert h003["right_probe"].eq("branch_only_output_loss").all()
    assert mix["left_probe"].eq("branch_only_output_loss").all()
    assert mix["right_probe"].eq("baseline_output_loss").all()
    assert h002["delta_holdout_mae"].lt(0).all()
    assert h003["delta_holdout_mae"].lt(0).all()


def test_summary_rejects_missing_holdout_correlation(tmp_path: Path):
    write_run(tmp_path, 1, 12345)
    path = expected_summary_path(tmp_path, 1, 12345)
    df = pd.read_csv(path).drop(columns=["final_holdout_pred_target_corr"])
    df.to_csv(path, index=False)
    try:
        load_all_runs(tmp_path, [1], [12345])
    except RuntimeError as exc:
        assert "final_holdout_pred_target_corr" in str(exc)
    else:
        raise AssertionError("missing holdout correlation should fail")


def test_decision_table_is_evidence_only_and_deterministic(tmp_path: Path):
    for episode in [1, 2]:
        for seed in [12345, 23456, 34567]:
            write_run(tmp_path, episode, seed)
    all_runs = load_all_runs(tmp_path, [1, 2], [12345, 23456, 34567])
    comparisons1 = pairwise_comparisons(all_runs)
    comparisons2 = pairwise_comparisons(all_runs)
    pd.testing.assert_frame_equal(comparisons1, comparisons2)

    decisions = decision_table(all_runs, comparisons1)
    assert set(decisions["hypothesis"]) == {
        "H002_supported",
        "H003_supported",
        "mix_objective_material",
    }
    assert decisions["caveat"].str.contains("Bellman regret is not included").any()
    assert not decisions.isna().any().any()


def test_slurm_validators_do_not_use_bool_identity_checks():
    for path in [
        Path("slurm/run_bp_recovery_probe_smoke.slurm"),
        Path("slurm/run_bp_recovery_probe_full_array.slurm"),
    ]:
        text = path.read_text()
        assert " is True" not in text
        assert " is False" not in text
        assert " is not True" not in text
        assert " is not False" not in text

    mix_flags = {
        "baseline_output_loss": np.bool_(True),
        "bias_recenter_output_loss": np.bool_(True),
        "branch_only_output_loss": np.bool_(False),
        "original_init_logit_loss": np.bool_(False),
    }
    assert bool(mix_flags["baseline_output_loss"])
    assert bool(mix_flags["bias_recenter_output_loss"])
    assert not bool(mix_flags["branch_only_output_loss"])
    assert not bool(mix_flags["original_init_logit_loss"])
