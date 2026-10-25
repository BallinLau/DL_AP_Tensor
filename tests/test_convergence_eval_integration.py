"""End-to-end integration tests of the convergence evaluation pipeline.

Runs a complete micro ``run_convergence_evaluation`` (CPU, synthetic combined
micro checkpoints + a tmp ``evaluation_bank_v1`` + scaled-down evaluator
configurations) and asserts the Prompt §13 output tree, the 14-section
Obsidian-compatible report (single-line ``$$..$$`` formulas, explicit reason
codes for every missing item), Evaluation-Only guarantees (parameter hash
unchanged, ``optimizer_steps == 0``), evaluator-failure containment and
repeat-run directory isolation.  No cachedir checkpoint is touched.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from analysis.economic_config import AnalysisEconomicConfig
from config import HyperParams
from evaluation.convergence.checkpoint_io import load_eval_checkpoint
from evaluation.convergence.comparator import ThresholdConfig
from evaluation.convergence.eval_bank import (
    EvalBankError,
    build_bank_a,
    build_bank_b,
    build_bank_c,
    build_bank_d,
    build_bank_e,
    compute_config_snapshot_hash,
    derive_macro_anchors,
    save_bank,
)
from evaluation.convergence.report import SECTION_TITLES
from evaluation.convergence import run as run_module
from evaluation.convergence.run import run_convergence_evaluation
from evaluation.convergence.figures import FIGURE_IDS
from experiments.run_utils import build_models
from models.policy_value import PolicyValueModel

CFG_HASH = compute_config_snapshot_hash(AnalysisEconomicConfig.from_current_config())

MICRO_OVERRIDES = {
    "bellman": {"parent_chunk_size": 4, "chunk_size": 64, "dense_grid_size": 9},
    "policy": {
        "b_points": 5,
        "z_points": 5,
        "i_points": 5,
        "objective_slice_points": 5,
        "representative_count": 3,
    },
    "macro": {
        "sdf_child_counts": (2, 4),
        "sdf_shock_bank_max_children": 4,
        "recursion_horizons": (1, 2, 5),
        "recursion_path_horizon": 4,
        "rollout_horizons": (1, 2, 5),
    },
    "simulation": {
        "variants": ("Baseline",),
        "capacity_extra_per_step": 20,
        "node_accounting_mode": "economic_node_ledger",
    },
}

REQUIRED_METRICS_FILES = (
    "bellman_metrics.csv",
    "value_drift.csv",
    "policy_drift.csv",
    "macro_metrics.csv",
    "sdf_metrics.csv",
    "simulation_metrics.csv",
    "distribution_metrics.csv",
    "convergence_status.json",
)
REQUIRED_TABLE_FILES = ("summary_table.csv", "episode_comparison.csv", "macro_state_table.csv")
REQUIRED_DIAGNOSTIC_FILES = ("runtime.json", "data_coverage.json", "failure_report.json")


# ---------------------------------------------------------------------------
# synthetic micro checkpoints and bank
# ---------------------------------------------------------------------------


def _bias_last_linear(module: torch.nn.Module, bias: float) -> None:
    last = None
    for child in module.modules():
        if isinstance(child, torch.nn.Linear):
            last = child
    assert last is not None
    with torch.no_grad():
        last.weight.mul_(0.0)
        last.bias.fill_(float(bias))


def _small_policy_value(seed: int = 0) -> PolicyValueModel:
    torch.manual_seed(seed)
    model = PolicyValueModel(
        base_state_dim=6,
        share_hidden_dims=[8],
        share_output_dim=8,
        q_head_dims=[8],
        q_parameterization="b_times_unit",
        p0_head_dims=[8],
        pi_head_dims=[8],
        bp0_head_dims=[8],
        bpi_head_dims=[8],
        barz_hidden_dims=[8],
        bari_hidden_dims=[8],
        dropout=0.0,
        i_grid_size=3,
        i_threshold=0.5,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.05)
    # Healthy-economy micro heads (mirrors the sim smoke test): survive and do
    # not invest, so the scaled-down rollout stays feasible.
    _bias_last_linear(model.v0_head, 3.0)
    _bias_last_linear(model.vi_head, -3.0)
    return model


def _hyperparams() -> HyperParams:
    hp = HyperParams()
    hp.bp_grid_min = 0.0
    hp.bp_grid_max = 1.0
    hp.bp_grid_coarse_size = 5
    hp.bp_grid_fine_size = 3
    hp.bp_grid_refine_enabled = True
    hp.bp_grid_quadratic_refine = False
    hp.bp_grid_parent_chunk_size = 64
    hp.bp_grid_candidate_chunk_size = 0
    hp.bp_grid_max_expanded_states = 4096
    hp.pv_use_clipped_m = True
    hp.pv_m_clamp_min = 0.7
    hp.pv_m_clamp_max = 1.3
    hp.pv_value_scale_mode = "none"
    hp.pv_value_scale_log_max = 20.0
    hp.pv_bellman_normalize_by_value_scale = False
    hp.q_recovery_normalization_mode = "asset_only"
    return hp


def _write_combined_checkpoint(path: Path, *, episode: int, seed: int = 0) -> Path:
    base = build_models(torch.device("cpu"))
    policy_value = _small_policy_value(seed=seed)
    hyperparams = _hyperparams()
    payload = {
        "models": {
            "policy_value": policy_value.state_dict(),
            "sdf_fc1": base["sdf_fc1"].state_dict(),
            "fc2": base["fc2"].state_dict(),
        },
        "hyperparams": dict(vars(hyperparams)),
        "config_snapshot": AnalysisEconomicConfig.from_current_config().to_dict(),
        "policy_value_model_spec": dict(policy_value.model_spec()),
        "value_parameterization": {
            "mode": "none",
            "scale_formula": "1",
            "bellman_normalization": False,
            "log_max": 20.0,
        },
        "episode": int(episode),
        "git_commit": "integration-test",
    }
    torch.save(payload, path)
    return path


def _synthetic_panel():
    firm_rows = [
        dict(path=0, t=0, branch=-1, ID="10", b=0.30, z=0.10, ETA=1.0, i=0.20, x=0.5, Hatcf=-2.05, LnKF=2.05, economic_node_id=0),
        dict(path=0, t=0, branch=-1, ID="11", b=0.40, z=-0.20, ETA=0.0, i=0.10, x=0.5, Hatcf=-1.95, LnKF=2.15, economic_node_id=0),
        dict(path=0, t=0, branch=0, ID="10", b=0.30, z=0.10, ETA=1.0, i=0.20, x=0.5, Hatcf=-9.05, LnKF=9.25, economic_node_id=0),
        dict(path=0, t=0, branch=0, ID="11", b=0.40, z=-0.20, ETA=0.0, i=0.10, x=0.5, Hatcf=-8.95, LnKF=9.45, economic_node_id=0),
        dict(path=0, t=1, branch=0, ID="10", b=0.31, z=0.11, ETA=1.0, i=0.21, x=0.6, Hatcf=-2.15, LnKF=4.15, economic_node_id=1),
        dict(path=0, t=1, branch=1, ID="10", b=0.32, z=0.12, ETA=0.0, i=0.22, x=0.6, Hatcf=-2.25, LnKF=4.25, economic_node_id=2),
        dict(path=0, t=1, branch=0, ID="11", b=0.41, z=-0.21, ETA=1.0, i=0.11, x=0.6, Hatcf=-2.35, LnKF=4.35, economic_node_id=1),
        dict(path=0, t=1, branch=1, ID="11", b=0.42, z=-0.22, ETA=0.0, i=0.12, x=0.6, Hatcf=-2.45, LnKF=4.45, economic_node_id=2),
        dict(path=0, t=1, branch=-1, ID="10", b=0.33, z=0.13, ETA=1.0, i=0.23, x=0.6, Hatcf=-1.85, LnKF=2.25, economic_node_id=3),
        dict(path=0, t=1, branch=-1, ID="11", b=0.43, z=-0.23, ETA=0.0, i=0.13, x=0.6, Hatcf=-1.75, LnKF=2.35, economic_node_id=3),
    ]
    macro_rows = [
        dict(path=0, t=0, branch=-1, K=1.0, C=0.5, LnK=8.0, Hatc=-5.0, n_firms=2, M=1.0, x=0.5, hatcf=-3.0, lnkf=6.0, economic_node_id=0),
        dict(path=0, t=0, branch=0, K=1.1, C=0.6, LnK=8.5, Hatc=-5.5, n_firms=2, M=1.0, x=0.5, hatcf=-3.5, lnkf=6.5, economic_node_id=0),
        dict(path=0, t=1, branch=-1, K=1.2, C=0.7, LnK=8.1, Hatc=-4.9, n_firms=2, M=1.0, x=0.6, hatcf=-3.1, lnkf=6.1, economic_node_id=3),
        dict(path=0, t=1, branch=0, K=1.3, C=0.8, LnK=8.6, Hatc=-5.4, n_firms=2, M=1.0, x=0.6, hatcf=-3.4, lnkf=6.4, economic_node_id=1),
        dict(path=0, t=1, branch=1, K=1.4, C=0.9, LnK=8.7, Hatc=-5.3, n_firms=2, M=1.0, x=0.6, hatcf=-3.3, lnkf=6.3, economic_node_id=2),
        dict(path=0, t=2, branch=-1, K=1.5, C=1.0, LnK=8.2, Hatc=-4.8, n_firms=2, M=1.0, x=0.7, hatcf=-3.2, lnkf=6.2, economic_node_id=6),
        dict(path=1, t=0, branch=-1, K=2.0, C=1.5, LnK=9.0, Hatc=-6.0, n_firms=1, M=1.0, x=-0.2, hatcf=-4.0, lnkf=7.0, economic_node_id=0),
        dict(path=1, t=1, branch=-1, K=2.1, C=1.6, LnK=9.1, Hatc=-5.9, n_firms=1, M=1.0, x=-0.1, hatcf=-4.1, lnkf=7.1, economic_node_id=3),
        dict(path=1, t=2, branch=-1, K=2.2, C=1.7, LnK=9.2, Hatc=-5.8, n_firms=1, M=1.0, x=0.0, hatcf=-4.2, lnkf=7.2, economic_node_id=6),
    ]
    return pd.DataFrame(firm_rows), pd.DataFrame(macro_rows)


def _build_bank_root(bank_root: Path, *, seed: int = 7) -> Path:
    firm_frame, macro_frame = _synthetic_panel()
    bank_a = build_bank_a(firm_frame, macro_frame, seed=seed, config_hash=CFG_HASH)
    bank_b = build_bank_b(bank_a, seed=seed + 1, config_hash=CFG_HASH)
    bank_d = build_bank_d(macro_frame, seed=seed + 3, config_hash=CFG_HASH)
    anchors = derive_macro_anchors(bank_d)
    bank_c = build_bank_c(seed=seed + 2, macro_anchors=anchors, config_hash=CFG_HASH)
    bank_e = build_bank_e(
        seed=seed + 4,
        n_firms=4,
        n_paths=2,
        horizon=3,
        economic_config=AnalysisEconomicConfig.from_current_config(),
        config_hash=CFG_HASH,
    )
    # Healthy-economy initial macro state for the scaled-down rollout.
    for variant in bank_e["variants"].values():
        n_rows = int(torch.as_tensor(variant["x0"]).numel())
        variant["x0"] = torch.full((n_rows,), 1.0)
        variant["z0"] = torch.zeros(n_rows)
    manifest = {
        "bank_version": "evaluation_bank_v1",
        "evaluation_code_commit": "integration-test",
        "config_snapshot_hash": CFG_HASH,
        "seeds": {
            "bank_a": seed,
            "bank_b": seed + 1,
            "bank_c": seed + 2,
            "bank_d": seed + 3,
            "bank_e": seed + 4,
        },
        "source": {
            "kind": "reference-policy-distribution",
            "checkpoint_ref": "integration-test",
            "train_lineage_exclusion": ["0::11"],
        },
    }
    save_bank(
        bank_root,
        {"A": bank_a, "B": bank_b, "C": bank_c, "D": bank_d, "E": bank_e},
        manifest,
    )
    return bank_root


def _make_checkpoints(tmp_path: Path, episodes: Sequence[int] = (3, 4)):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for offset, episode in enumerate(episodes):
        paths.append(
            _write_combined_checkpoint(
                checkpoint_dir / f"ep{episode}_combined.pt", episode=episode, seed=11 + offset
            )
        )
    return paths


def _run_micro(tmp_path: Path, *, run_id: str = "micro", episodes=(3, 4), mode: str = "all"):
    bank_root = tmp_path / "bank"
    if not (bank_root / "bank_manifest.json").is_file():
        _build_bank_root(bank_root)
    checkpoint_paths = _make_checkpoints(tmp_path, episodes)
    return run_convergence_evaluation(
        checkpoint_paths=checkpoint_paths,
        bank_root=bank_root,
        output_root=tmp_path / "eval_out",
        run_id=run_id,
        mode=mode,
        device="cpu",
        seed=7,
        chunk_size=4,
        rollout_horizon=3,
        threshold_config=None,
        eval_config_overrides=MICRO_OVERRIDES,
    )


# ---------------------------------------------------------------------------
# Test A — full pipeline output tree + report contract + Evaluation-Only
# ---------------------------------------------------------------------------


def test_end_to_end_micro_run_writes_full_output_tree(tmp_path):
    summary = _run_micro(tmp_path, run_id="micro_all", episodes=(3, 4), mode="all")
    run_dir = Path(summary["run_dir"])

    # ---- Prompt §13 output tree ----------------------------------------
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "bank" / "bank_manifest.json").is_file()
    for name in REQUIRED_METRICS_FILES:
        assert (run_dir / "metrics" / name).is_file(), name
    for name in REQUIRED_TABLE_FILES:
        assert (run_dir / "tables" / name).is_file(), name
    for name in REQUIRED_DIAGNOSTIC_FILES:
        assert (run_dir / "diagnostics" / name).is_file(), name
    assert (run_dir / "report" / "convergence_report.md").is_file()
    figures_dir = run_dir / "figures"
    assert figures_dir.is_dir()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    figures = manifest["figures"]["figures"]
    assert sorted(figures.keys()) == sorted(FIGURE_IDS)
    for figure_id, record in figures.items():
        if record["status"] == "ok":
            assert Path(record["path"]).is_file(), figure_id
            assert (figures_dir / Path(record["path"]).name).is_file()
        else:
            assert record["missing_reason"], figure_id

    # ---- Evaluation-Only (Test 10, system level) ------------------------
    assert manifest["param_hash"]["all_unchanged"] is True
    assert manifest["param_hash"]["optimizer_steps"] == 0
    for episode_key, record in manifest["param_hash"]["per_episode"].items():
        assert record["unchanged"] is True, episode_key
        assert record["optimizer_steps"] == 0
        assert record["before"] == record["after"]
    assert summary["param_hashes_unchanged"] is True
    assert summary["optimizer_steps"] == 0

    runtime = json.loads((run_dir / "diagnostics" / "runtime.json").read_text(encoding="utf-8"))
    assert runtime["evaluation_wall_time_seconds"] > 0
    assert runtime["optimizer_steps"] == 0
    teacher = runtime["grid_teacher"]
    assert teacher["n_grid_teachers"] >= 1
    assert teacher["forward_stats_total"].get("bp_model_forward_calls", 0) >= 1
    cache = runtime["cache"]
    assert cache["cache_items"] >= 1
    assert cache["cache_hits"] >= 1
    assert cache["hit_rate"] is not None

    coverage = json.loads((run_dir / "diagnostics" / "data_coverage.json").read_text(encoding="utf-8"))
    assert set(coverage["banks"].keys()) == {"A", "B", "C", "D", "E"}
    for bank_key, entry in coverage["banks"].items():
        assert entry["n_rows"] is not None, bank_key
        assert entry["participating_episodes"] == [3, 4]

    # ---- report contract -----------------------------------------------
    report = (run_dir / "report" / "convergence_report.md").read_text(encoding="utf-8")
    section_headers = re.findall(r"^## (\d+)\. (.+)$", report, flags=re.MULTILINE)
    assert [int(number) for number, _ in section_headers] == list(range(1, 15))
    assert [title for _, title in section_headers] == list(SECTION_TITLES)

    # every formula lives on a single line (Obsidian contract)
    assert report.count("$$") >= 2
    assert report.count("$$") % 2 == 0
    for line in report.splitlines():
        assert line.count("$$") % 2 == 0, f"multi-line $$ block: {line!r}"
    formula_blocks = re.findall(r"\$\$(.+?)\$\$", report, flags=re.DOTALL)
    assert formula_blocks
    assert all("\n" not in block for block in formula_blocks)

    # figures/tables are referenced by relative path, missing items carry codes
    for figure_id, record in figures.items():
        if record["status"] == "ok":
            assert f"figures/{Path(record['path']).name}" in report, figure_id
        else:
            assert record["missing_reason"] in report, figure_id
    for item in manifest["missing_items"]:
        assert item["reason_code"] in report, item
    for name in ("metrics/value_drift.csv", "tables/macro_state_table.csv", "bank/bank_manifest.json"):
        assert name in report

    # ---- convergence status artifacts ----------------------------------
    status = json.loads(
        (run_dir / "metrics" / "convergence_status.json").read_text(encoding="utf-8")
    )
    assert status["convergence_status"] in {
        "CONVERGED_CANDIDATE",
        "NOT_CONVERGED",
        "INSUFFICIENT_EVIDENCE",
    }
    assert status["episodes"] == [3, 4]
    gates = status["episode_status"][-1]["gates"]
    assert set(gates.keys()) == {"hard_economic", "firm", "macro", "outer"}
    all_statuses = {
        gate["status"]
        for group in gates.values()
        for gate in group.values()
    }
    assert all_statuses <= {"PASS", "FAIL", "INSUFFICIENT_EVIDENCE", "THRESHOLD_NOT_CONFIGURED"}

    # absolute errors + sample sizes survive into the metric tables
    value_drift = pd.read_csv(run_dir / "metrics" / "value_drift.csv")
    assert "p0_physical_mae" in value_drift.columns
    assert "p0_physical_n" in value_drift.columns
    bellman = pd.read_csv(run_dir / "metrics" / "bellman_metrics.csv")
    assert "fresh_p0_normalized_mae" in bellman.columns

    macro_state = pd.read_csv(run_dir / "tables" / "macro_state_table.csv")
    assert {"episode", "macro_state", "status", "reason_code"} <= set(macro_state.columns)
    assert set(macro_state["macro_state"]) == {"recession", "normal", "expansion"}


# ---------------------------------------------------------------------------
# Test B — single evaluator failure never kills the run
# ---------------------------------------------------------------------------


def test_single_evaluator_failure_is_recorded_and_run_continues(tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("injected evaluator failure")

    monkeypatch.setattr(run_module, "evaluate_firm_policy", _boom)
    summary = _run_micro(tmp_path, run_id="micro_fail", episodes=(3, 4), mode="firm")
    run_dir = Path(summary["run_dir"])
    assert run_dir.is_dir()
    assert summary["n_failures"] >= 1

    failure_report = json.loads(
        (run_dir / "diagnostics" / "failure_report.json").read_text(encoding="utf-8")
    )
    entries = [item for item in failure_report["failures"] if item["evaluator"] == "policy"]
    assert entries, failure_report
    for entry in entries:
        assert entry["reason_code"] == "EVALUATOR_FAILED"
        assert "injected evaluator failure" in entry["detail"]

    # the run still produced the report and the missing group carries a code
    assert (run_dir / "report" / "convergence_report.md").is_file()
    policy_drift = pd.read_csv(run_dir / "metrics" / "policy_drift.csv")
    assert (policy_drift["reason_code"].astype(str).str.contains("MISSING_POLICY_RESULT")).any()
    report = (run_dir / "report" / "convergence_report.md").read_text(encoding="utf-8")
    assert "MISSING_POLICY_RESULT" in report
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["param_hash"]["all_unchanged"] is True
    assert manifest["param_hash"]["optimizer_steps"] == 0


# ---------------------------------------------------------------------------
# Test C — repeated runs get independent directories, never overwrite
# ---------------------------------------------------------------------------


def test_repeat_run_creates_independent_directory(tmp_path):
    first = _run_micro(tmp_path, run_id="micro_repeat", episodes=(3,), mode="firm")
    first_dir = Path(first["run_dir"])
    first_manifest = (first_dir / "manifest.json").read_bytes()
    second = _run_micro(tmp_path, run_id="micro_repeat", episodes=(3,), mode="firm")
    second_dir = Path(second["run_dir"])

    assert first_dir != second_dir
    assert first_dir.is_dir() and second_dir.is_dir()
    assert (first_dir / "manifest.json").read_bytes() == first_manifest
    assert (second_dir / "manifest.json").is_file()
    # old artifacts survive untouched
    for name in REQUIRED_METRICS_FILES:
        assert (first_dir / "metrics" / name).is_file(), name


# ---------------------------------------------------------------------------
# Test D — --build-bank builds evaluation_bank_v1 and refuses to overwrite
# ---------------------------------------------------------------------------


def test_build_bank_from_source_panels_and_refuses_existing_root(tmp_path):
    firm_frame, macro_frame = _synthetic_panel()
    firm_panel = tmp_path / "firm_panel.pkl"
    macro_panel = tmp_path / "macro_panel.pkl"
    firm_frame.to_pickle(firm_panel)
    macro_frame.to_pickle(macro_panel)
    bank_root = tmp_path / "built_bank"

    summary = run_convergence_evaluation(
        checkpoint_paths=_make_checkpoints(tmp_path, (3,)),
        bank_root=bank_root,
        output_root=tmp_path / "eval_out",
        run_id="micro_buildbank",
        mode="firm",
        device="cpu",
        seed=7,
        chunk_size=4,
        rollout_horizon=3,
        build_bank={"firm_panel": str(firm_panel), "macro_panel": str(macro_panel)},
        eval_config_overrides=MICRO_OVERRIDES,
    )
    bank_manifest = json.loads((bank_root / "bank_manifest.json").read_text(encoding="utf-8"))
    assert bank_manifest["bank_version"] == "evaluation_bank_v1"
    assert bank_manifest["source"]["kind"] == "reference-policy-distribution"
    assert set(bank_manifest["banks"].keys()) == {"A", "B", "C", "D", "E"}
    assert Path(summary["run_dir"]).is_dir()

    with pytest.raises(EvalBankError):
        run_convergence_evaluation(
            checkpoint_paths=_make_checkpoints(tmp_path, (3,)),
            bank_root=bank_root,
            output_root=tmp_path / "eval_out",
            run_id="micro_buildbank_again",
            mode="firm",
            device="cpu",
            seed=7,
            chunk_size=4,
            rollout_horizon=3,
            build_bank={"firm_panel": str(firm_panel), "macro_panel": str(macro_panel)},
            eval_config_overrides=MICRO_OVERRIDES,
        )


# ---------------------------------------------------------------------------
# Test E — CLI dispatch (no evaluation executed)
# ---------------------------------------------------------------------------


def test_cli_dispatches_to_run_convergence_evaluation(tmp_path, monkeypatch):
    captured = {}

    def _stub(**kwargs):
        captured.update(kwargs)
        return {"run_dir": "unused", "manifest_path": "unused", "report_path": "unused",
                "episodes": [], "convergence_status": "INSUFFICIENT_EVIDENCE",
                "converged_candidate": False, "n_failures": 0, "figures_ok": 0,
                "figures_skipped": 0, "param_hashes_unchanged": True, "optimizer_steps": 0}

    monkeypatch.setattr(run_module, "run_convergence_evaluation", _stub)
    threshold_path = tmp_path / "thresholds.json"
    threshold_path.write_text(json.dumps({"firm_p_bellman_residual": 0.1}), encoding="utf-8")
    code = run_module.main(
        [
            "--checkpoint", "ep3_combined.pt,ep4_combined.pt",
            "--bank-root", str(tmp_path / "bank"),
            "--output-root", str(tmp_path / "out"),
            "--run-id", "cli",
            "--mode", "firm",
            "--device", "cpu",
            "--seed", "11",
            "--chunk-size", "8",
            "--rollout-horizon", "5",
            "--threshold-config", str(threshold_path),
            "--build-bank", "firm.pkl,macro.pkl",
        ]
    )
    assert code == 0
    assert captured["checkpoint_paths"] == ["ep3_combined.pt", "ep4_combined.pt"]
    assert captured["mode"] == "firm"
    assert captured["seed"] == 11
    assert captured["chunk_size"] == 8
    assert captured["rollout_horizon"] == 5
    assert captured["build_bank"] == {"firm_panel": "firm.pkl", "macro_panel": "macro.pkl"}
    assert isinstance(captured["threshold_config"], ThresholdConfig)
    assert captured["threshold_config"].get("firm_p_bellman_residual") == pytest.approx(0.1)
