"""Core tests for the equilibrium convergence evaluator package.

Covers Prompt Test 1 (checkpoint semantics), Test 2 (EvalBank reproducibility),
Test 3 (panel timing/pairing) and Test 10 (Evaluation-Only parameter hash
guard).  Everything runs on CPU with synthetic micro checkpoints in tmp_path —
no cachedir checkpoint is required.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from analysis.economic_config import AnalysisEconomicConfig
from config import HyperParams
from evaluation.convergence.checkpoint_io import (
    EvalCheckpointError,
    check_semantic_compatibility,
    count_optimizer_steps,
    load_eval_checkpoint,
    param_hash_guard,
    semantic_record,
)
from evaluation.convergence.eval_bank import (
    EvalBankError,
    attach_macro_context,
    build_bank_a,
    build_bank_b,
    build_bank_c,
    build_bank_d,
    build_bank_e,
    build_transition_for_checkpoint,
    compute_config_snapshot_hash,
    derive_macro_anchors,
    hash_split_lineages,
    load_bank,
    pair_firm_parent_child,
    save_bank,
    select_parent_view_rows,
)
from models.policy_value import PolicyValueModel
from experiments.run_utils import build_models


CFG_HASH = compute_config_snapshot_hash(AnalysisEconomicConfig.from_current_config())


# ---------------------------------------------------------------------------
# synthetic micro checkpoints
# ---------------------------------------------------------------------------


def _small_policy_value(q_parameterization: str = "b_times_unit", seed: int = 0) -> PolicyValueModel:
    torch.manual_seed(seed)
    model = PolicyValueModel(
        base_state_dim=6,
        share_hidden_dims=[8],
        share_output_dim=8,
        q_head_dims=[8],
        q_parameterization=q_parameterization,
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
    return model


def _write_combined_checkpoint(
    path: Path,
    *,
    episode: int = 3,
    q_parameterization: str = "b_times_unit",
    drop_spec_keys=(),
    drop_hyperparam_keys=(),
    value_mode: str = "none",
    config_overrides=None,
    git_commit: str = "testhash",
    seed: int = 0,
):
    base = build_models(torch.device("cpu"))
    policy_value = _small_policy_value(q_parameterization, seed=seed)

    hyperparams_dict = dict(vars(HyperParams()))
    for key in drop_hyperparam_keys:
        hyperparams_dict.pop(key, None)
    if "q_recovery_normalization_mode" not in drop_hyperparam_keys:
        if hyperparams_dict.get("q_recovery_normalization_mode") is None:
            hyperparams_dict["q_recovery_normalization_mode"] = "asset_only"
    if value_mode == "exp_xz":
        hyperparams_dict["pv_value_scale_mode"] = "exp_xz"

    spec = dict(policy_value.model_spec())
    for key in drop_spec_keys:
        spec.pop(key, None)

    log_max = float(hyperparams_dict["pv_value_scale_log_max"])
    mode = str(hyperparams_dict["pv_value_scale_mode"])
    value_parameterization = {
        "mode": mode,
        "scale_formula": f"1+exp(clamp(x+z,max={log_max:g}))" if mode == "exp_xz" else "1",
        "bellman_normalization": bool(hyperparams_dict["pv_bellman_normalize_by_value_scale"]),
        "log_max": log_max,
    }

    config_snapshot = AnalysisEconomicConfig.from_current_config().to_dict()
    for key, value in (config_overrides or {}).items():
        config_snapshot[key] = float(value)

    payload = {
        "models": {
            "policy_value": policy_value.state_dict(),
            "sdf_fc1": base["sdf_fc1"].state_dict(),
            "fc2": base["fc2"].state_dict(),
        },
        "hyperparams": hyperparams_dict,
        "config_snapshot": config_snapshot,
        "policy_value_model_spec": spec,
        "value_parameterization": value_parameterization,
        "episode": int(episode),
        "git_commit": git_commit,
    }
    torch.save(payload, path)
    return payload


# ---------------------------------------------------------------------------
# synthetic firm/macro panel (dual-view rows, known parent-child pairing)
# ---------------------------------------------------------------------------


def _synthetic_panel():
    # Values are log-scale macro quantities, so they stay in a numerically sane
    # range while the three column families (firm Hatcf/LnKF beliefs, macro
    # Hatc/LnK realized, macro hatcf/lnkf beliefs) remain clearly distinguishable.
    firm_rows = [
        # parent view (t=0, branch=-1)
        dict(path=0, t=0, branch=-1, ID="10", b=0.30, z=0.10, ETA=1.0, i=0.20, x=0.5, Hatcf=-2.05, LnKF=2.05, economic_node_id=0),
        dict(path=0, t=0, branch=-1, ID="11", b=0.40, z=-0.20, ETA=0.0, i=0.10, x=0.5, Hatcf=-1.95, LnKF=2.15, economic_node_id=0),
        # dual-view child rows of the same node (pre-exit), distinct belief columns
        dict(path=0, t=0, branch=0, ID="10", b=0.30, z=0.10, ETA=1.0, i=0.20, x=0.5, Hatcf=-9.05, LnKF=9.25, economic_node_id=0),
        dict(path=0, t=0, branch=0, ID="11", b=0.40, z=-0.20, ETA=0.0, i=0.10, x=0.5, Hatcf=-8.95, LnKF=9.45, economic_node_id=0),
        # child rows at t=1 for both branches
        dict(path=0, t=1, branch=0, ID="10", b=0.31, z=0.11, ETA=1.0, i=0.21, x=0.6, Hatcf=-2.15, LnKF=4.15, economic_node_id=1),
        dict(path=0, t=1, branch=1, ID="10", b=0.32, z=0.12, ETA=0.0, i=0.22, x=0.6, Hatcf=-2.25, LnKF=4.25, economic_node_id=2),
        dict(path=0, t=1, branch=0, ID="11", b=0.41, z=-0.21, ETA=1.0, i=0.11, x=0.6, Hatcf=-2.35, LnKF=4.35, economic_node_id=1),
        dict(path=0, t=1, branch=1, ID="11", b=0.42, z=-0.22, ETA=0.0, i=0.12, x=0.6, Hatcf=-2.45, LnKF=4.45, economic_node_id=2),
        # parent view at t=1 (no t=2 children -> pairing must flag them unmatched)
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


def _build_all_banks(seed: int):
    firm_frame, macro_frame = _synthetic_panel()
    bank_a = build_bank_a(firm_frame, macro_frame, seed=seed, config_hash=CFG_HASH)
    bank_b = build_bank_b(bank_a, seed=seed + 1, config_hash=CFG_HASH)
    bank_d = build_bank_d(macro_frame, seed=seed + 3, config_hash=CFG_HASH)
    anchors = derive_macro_anchors(bank_d)
    bank_c = build_bank_c(seed=seed + 2, macro_anchors=anchors, config_hash=CFG_HASH)
    bank_e = build_bank_e(
        seed=seed + 4,
        n_firms=8,
        n_paths=2,
        horizon=5,
        economic_config=AnalysisEconomicConfig.from_current_config(),
        config_hash=CFG_HASH,
    )
    return {"A": bank_a, "B": bank_b, "C": bank_c, "D": bank_d, "E": bank_e}, anchors


def _manifest() -> dict:
    return {
        "bank_version": "evaluation_bank_v1",
        "evaluation_code_commit": "testcommit",
        "config_snapshot_hash": CFG_HASH,
        "source": {
            "kind": "reference-policy-distribution",
            "checkpoint_ref": "deadbeef",
            "train_lineage_exclusion": ["0::11"],
        },
    }


# ---------------------------------------------------------------------------
# Test 1 — checkpoint semantics
# ---------------------------------------------------------------------------


def test_load_eval_checkpoint_semantic_record(tmp_path):
    ckpt_path = tmp_path / "ep3_combined.pt"
    _write_combined_checkpoint(ckpt_path, episode=3)

    cp = load_eval_checkpoint(ckpt_path)

    assert set(cp.models.keys()) >= {"policy_value", "sdf_fc1", "fc2"}
    assert cp.episode == 3
    assert cp.checkpoint_path == str(ckpt_path)
    assert set(cp.param_hashes.keys()) == set(cp.models.keys())
    for digest in cp.param_hashes.values():
        assert len(digest) == 64

    record = semantic_record(cp)
    assert record["checkpoint_sha256"]
    assert record["checkpoint_format"] == "combined"
    assert record["episode"] == 3
    assert record["git_commit"] == "testhash"
    assert record["q_parameterization"] == "b_times_unit"
    assert record["q_parameterization_source"] == "checkpoint_spec"
    assert record["q_parameterization_assumed_legacy"] is False
    assert "q_recovery_normalization_mode" in record["q_semantics_recorded_fields"]
    assert record["q_recovery_normalization_mode"] == "asset_only"
    assert record["q_recovery_normalization_mode_assumed_legacy"] is False
    assert record["value_parameterization"]["mode"] == "none"
    assert record["value_parameterization"]["scale_formula"] == "1"
    assert record["value_parameterization"]["log_max"] == 20.0
    assert set(record["value_parameterization"].keys()) == {
        "mode",
        "scale_formula",
        "bellman_normalization",
        "log_max",
    }
    assert record["economic_config"] == AnalysisEconomicConfig.from_current_config().to_dict()
    assert record["optimizer_steps"] == 0
    assert record["evaluation_only"] is True
    assert record["policy_value_model_spec"]["q_parameterization"] == "b_times_unit"


def test_legacy_missing_fields_are_annotated(tmp_path):
    ckpt_path = tmp_path / "ep1_combined.pt"
    _write_combined_checkpoint(
        ckpt_path,
        episode=1,
        drop_spec_keys=("q_parameterization",),
        drop_hyperparam_keys=("q_recovery_normalization_mode",),
    )

    cp = load_eval_checkpoint(ckpt_path)
    record = semantic_record(cp)

    assert record["q_parameterization"] == "b_times_unit"
    assert record["q_parameterization_source"] == "q_parameterization_assumed_legacy"
    assert record["q_parameterization_assumed_legacy"] is True
    assert record["q_recovery_normalization_mode"] == "legacy_b_times_unit"
    assert record["q_recovery_normalization_mode_source"] == (
        "q_recovery_normalization_mode_assumed_legacy"
    )
    assert record["q_recovery_normalization_mode_assumed_legacy"] is True
    assert "q_recovery_normalization_mode" not in record["q_semantics_recorded_fields"]


def test_bare_state_dict_is_rejected_with_reason_code(tmp_path):
    payload = _write_combined_checkpoint(tmp_path / "ep2_combined.pt", episode=2)
    bare_path = tmp_path / "ep2_policy_value.pt"
    torch.save(payload["models"]["policy_value"], bare_path)

    with pytest.raises(EvalCheckpointError) as excinfo:
        load_eval_checkpoint(bare_path)
    assert excinfo.value.reason_code == "NOT_COMBINED_CHECKPOINT"

    no_models_path = tmp_path / "ep2_no_models.pt"
    torch.save({"policy_value": payload["models"]["policy_value"]}, no_models_path)
    with pytest.raises(EvalCheckpointError) as excinfo:
        load_eval_checkpoint(no_models_path)
    assert excinfo.value.reason_code == "NOT_COMBINED_CHECKPOINT"

    missing_model_path = tmp_path / "ep2_missing_sdf.pt"
    torch.save({"models": {"policy_value": payload["models"]["policy_value"]}}, missing_model_path)
    with pytest.raises(EvalCheckpointError) as excinfo:
        load_eval_checkpoint(missing_model_path)
    assert excinfo.value.reason_code == "MISSING_REQUIRED_MODEL"


def test_check_semantic_compatibility_reports_field_diffs(tmp_path):
    _write_combined_checkpoint(tmp_path / "ep3_combined.pt", episode=3, seed=1)
    _write_combined_checkpoint(tmp_path / "ep4_combined.pt", episode=4, seed=2)
    _write_combined_checkpoint(
        tmp_path / "ep5_combined.pt",
        episode=5,
        seed=3,
        q_parameterization="direct",
        value_mode="exp_xz",
        config_overrides={"RHO_X": 0.123456},
    )

    cp3 = load_eval_checkpoint(tmp_path / "ep3_combined.pt")
    cp4 = load_eval_checkpoint(tmp_path / "ep4_combined.pt")
    cp5 = load_eval_checkpoint(tmp_path / "ep5_combined.pt")

    same = check_semantic_compatibility([cp3, cp4])
    assert same["compatible"] is True
    assert same["diffs"] == {}
    assert same["labels"] == [cp3.metadata["label"], cp4.metadata["label"]]
    assert "q_parameterization" in same["checked_fields"]
    assert "value_parameterization.mode" in same["checked_fields"]
    assert "economic_config.RHO_X" in same["checked_fields"]
    assert "policy_value_model_spec" in same["checked_fields"]

    diff = check_semantic_compatibility([cp3, cp4, cp5])
    assert diff["compatible"] is False
    assert diff["diffs"]["q_parameterization"][cp5.metadata["label"]] == "direct"
    assert diff["diffs"]["q_parameterization"][cp3.metadata["label"]] == "b_times_unit"
    assert diff["diffs"]["value_parameterization.mode"][cp5.metadata["label"]] == "exp_xz"
    assert diff["diffs"]["value_parameterization.scale_formula"][cp5.metadata["label"]] == (
        "1+exp(clamp(x+z,max=20))"
    )
    assert "economic_config.RHO_X" in diff["diffs"]
    assert "economic_config.SIGMA_X" not in diff["diffs"]
    assert diff["fields"]["economic_config.SIGMA_X"]["consistent"] is True

    with pytest.raises(EvalCheckpointError) as excinfo:
        check_semantic_compatibility([cp3, cp5], strict=True)
    assert excinfo.value.reason_code == "SEMANTIC_MISMATCH"


def test_count_optimizer_steps_is_zero_and_recorded(tmp_path):
    assert count_optimizer_steps() == 0
    ckpt_path = tmp_path / "ep6_combined.pt"
    _write_combined_checkpoint(ckpt_path, episode=6)
    cp = load_eval_checkpoint(ckpt_path)
    record = semantic_record(cp)
    assert record["optimizer_steps"] == 0
    assert record["evaluation_only"] is True


# ---------------------------------------------------------------------------
# Test 10 — Evaluation-Only parameter hash guard
# ---------------------------------------------------------------------------


def test_param_hash_guard_detects_parameter_mutation(tmp_path):
    ckpt_path = tmp_path / "ep7_combined.pt"
    _write_combined_checkpoint(ckpt_path, episode=7)
    cp = load_eval_checkpoint(ckpt_path)

    with param_hash_guard(cp):
        pass

    model = cp.models["policy_value"]
    with pytest.raises(EvalCheckpointError) as excinfo:
        with param_hash_guard(cp):
            with torch.no_grad():
                first_parameter = next(iter(model.parameters()))
                first_parameter.add_(0.123)
    assert excinfo.value.reason_code == "PARAM_HASH_MISMATCH"


# ---------------------------------------------------------------------------
# Test 2 — EvalBank reproducibility / persistence
# ---------------------------------------------------------------------------


def test_banks_are_byte_identical_for_same_seed_and_differ_across_seeds(tmp_path):
    banks_a, _ = _build_all_banks(seed=100)
    banks_b, _ = _build_all_banks(seed=100)
    banks_c, _ = _build_all_banks(seed=200)

    root_a = tmp_path / "bank_run_a"
    root_b = tmp_path / "bank_run_b"
    root_c = tmp_path / "bank_run_c"
    manifest_a = save_bank(root_a, banks_a, _manifest())["manifest"]
    manifest_b = save_bank(root_b, banks_b, _manifest())["manifest"]
    manifest_c = save_bank(root_c, banks_c, _manifest())["manifest"]

    for key in ("A", "B", "C", "D", "E"):
        assert manifest_a["banks"][key]["sha256"] == manifest_b["banks"][key]["sha256"], key
        assert manifest_a["banks"][key]["content_sha256"] == manifest_b["banks"][key]["content_sha256"], key
        assert manifest_a["banks"][key]["sha256"] != manifest_c["banks"][key]["sha256"], key

    loaded = load_bank(root_a)
    assert set(loaded["banks"].keys()) == {"A", "B", "C", "D", "E"}
    assert loaded["banks"]["A"]["shock_bank_sha256"] == banks_a["A"]["shock_bank_sha256"]
    assert torch.equal(loaded["banks"]["A"]["parent_states"], banks_a["A"]["parent_states"])


def test_save_bank_refuses_existing_root_and_version_upgrade_warns(tmp_path):
    banks, _ = _build_all_banks(seed=100)
    root = tmp_path / "bank_root"
    first = save_bank(root, banks, _manifest())
    assert first["incomparable"] is False

    with pytest.raises(EvalBankError) as excinfo:
        save_bank(root, banks, _manifest())
    assert excinfo.value.reason_code == "BANK_ROOT_EXISTS"

    with pytest.raises(EvalBankError) as excinfo:
        save_bank(root, banks, _manifest(), version="evaluation_bank_v1")
    assert excinfo.value.reason_code == "BANK_VERSION_NOT_UPGRADED"

    upgraded_manifest = _manifest()
    upgraded_manifest["bank_version"] = "evaluation_bank_v2"
    upgraded = save_bank(root, banks, upgraded_manifest, version="evaluation_bank_v2")
    assert upgraded["incomparable"] is True
    assert upgraded["warnings"]
    assert upgraded["manifest"]["incomparable_with_previous_version"] is True
    assert upgraded["manifest"]["bank_version"] == "evaluation_bank_v2"


def test_manifest_fields_are_complete(tmp_path):
    banks, _ = _build_all_banks(seed=100)
    root = tmp_path / "bank_root"
    manifest = save_bank(root, banks, _manifest())["manifest"]

    for key in ("bank_version", "created_utc", "evaluation_code_commit", "seeds",
                "config_snapshot_hash", "source", "banks"):
        assert key in manifest, key
    assert manifest["bank_version"] == "evaluation_bank_v1"
    assert manifest["config_snapshot_hash"] == CFG_HASH
    for seed_key in ("bank_a", "bank_b", "bank_c", "bank_d", "bank_e"):
        assert seed_key in manifest["seeds"]
    for source_key in ("kind", "checkpoint_ref", "train_lineage_exclusion"):
        assert source_key in manifest["source"]
    for bank_key in ("A", "B", "C", "D", "E"):
        entry = manifest["banks"][bank_key]
        for field in ("file", "sha256", "n_rows", "eta_share", "source_share", "columns", "schema_version"):
            assert field in entry, (bank_key, field)
        assert entry["n_rows"] > 0
        if bank_key in ("A", "B", "C"):
            assert sum(entry["eta_share"].values()) == pytest.approx(1.0)

    assert set(banks["A"]["columns"]) == {"b", "z", "ETA", "i", "x", "Hatcf", "LnKF"}
    assert banks["A"]["tags"] == ["reference-policy-distribution"]
    assert banks["B"]["tags"] == ["counterfactual-eta-view"]
    assert banks["C"]["tags"] == ["out-of-distribution-stress"]
    assert banks["A"]["shock_bank_sha256"]
    assert banks["A"]["config_snapshot_hash"] == CFG_HASH
    assert banks["A"]["pairing_keys"]["path"]


def test_load_bank_verifies_file_hashes(tmp_path):
    banks, _ = _build_all_banks(seed=100)
    root = tmp_path / "bank_root"
    save_bank(root, banks, _manifest())

    tampered = root / "bank_a" / "parents.pt"
    with tampered.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(EvalBankError) as excinfo:
        load_bank(root)
    assert excinfo.value.reason_code == "BANK_FILE_HASH_MISMATCH"


def test_hash_split_lineages_is_deterministic_and_disjoint():
    lineages = [f"{path}::{firm}" for path in range(3) for firm in range(4)]
    first = hash_split_lineages(lineages, holdout_fraction=0.5, salt="unit-test")
    second = hash_split_lineages(lineages, holdout_fraction=0.5, salt="unit-test")
    assert first == second
    assert set(first["holdout"]).isdisjoint(first["excluded"])
    assert sorted(first["holdout"] + first["excluded"]) == sorted(lineages)


# ---------------------------------------------------------------------------
# Test 3 — panel timing / pairing / column semantics
# ---------------------------------------------------------------------------


def test_parent_view_filtering_prefers_branch_minus_one():
    firm_frame, _ = _synthetic_panel()
    parents = select_parent_view_rows(firm_frame)

    assert len(parents) == 4
    assert set(pd.to_numeric(parents["branch"])) == {-1}
    assert sorted(parents["Hatcf"].tolist()) == [-2.05, -1.95, -1.85, -1.75]
    # the dual-view child rows (branch=0) must never be selected
    assert -9.05 not in parents["Hatcf"].tolist()

    fallback = select_parent_view_rows(firm_frame.loc[firm_frame["branch"] != -1])
    assert set(pd.to_numeric(fallback["branch"])) == {0}


def test_pairing_keys_match_parent_child_rows():
    firm_frame, _ = _synthetic_panel()
    parents, pair_table = pair_firm_parent_child(firm_frame)

    assert len(parents) == 4
    matched = pair_table.loc[pair_table["matched"]]
    assert len(matched) == 4
    keys = {
        (int(row.path), str(row.ID), int(row.t), int(row.child_branch), int(row.child_economic_node_id))
        for row in matched.itertuples()
    }
    assert keys == {
        (0, "10", 0, 0, 1),
        (0, "10", 0, 1, 2),
        (0, "11", 0, 0, 1),
        (0, "11", 0, 1, 2),
    }

    # parents at t=1 have no keyed children at t=2: flagged unmatched, never
    # silently paired with row-adjacent rows.
    unmatched = pair_table.loc[~pair_table["matched"]]
    assert len(unmatched) == 4
    assert set(unmatched["t"].astype(int)) == {1}
    assert set(unmatched["child_t"].astype(int)) == {2}


def test_hatcf_lnkf_and_hatc_lnk_are_not_mixed():
    firm_frame, macro_frame = _synthetic_panel()
    bank_a = build_bank_a(firm_frame, macro_frame, seed=7, config_hash=CFG_HASH)

    states = bank_a["parent_states"]
    # firm-side belief states stay in the 7-dim state (columns 5/6)
    assert states[:, 5].tolist() == pytest.approx([-2.05, -1.95, -1.85, -1.75])
    assert states[:, 6].tolist() == pytest.approx([2.05, 2.15, 2.25, 2.35])
    # realized macro aggregates become the transition macro context (key join)
    assert bank_a["hatc_cal"].tolist() == pytest.approx([-5.0, -5.0, -4.9, -4.9])
    assert bank_a["lnk_cal"].tolist() == pytest.approx([8.0, 8.0, 8.1, 8.1])
    # macro belief columns must not leak into either place
    assert -3.0 not in states[:, 5].tolist()
    assert -3.0 not in bank_a["hatc_cal"].tolist()
    assert 6.0 not in bank_a["lnk_cal"].tolist()
    assert -5.0 not in states[:, 5].tolist()
    assert 8.0 not in states[:, 6].tolist()

    parents = select_parent_view_rows(firm_frame)
    attached = attach_macro_context(parents, macro_frame)
    assert attached["hatc_cal"].tolist() == pytest.approx([-5.0, -5.0, -4.9, -4.9])

    bank_d = build_bank_d(macro_frame, seed=7, config_hash=CFG_HASH)
    assert bank_d["hatcf_t"] == pytest.approx([-3.0, -3.1, -4.0, -4.1])
    assert bank_d["lnkf_t"] == pytest.approx([6.0, 6.1, 7.0, 7.1])
    assert bank_d["hatc_t"] == pytest.approx([-5.0, -4.9, -6.0, -5.9])
    assert bank_d["lnk_t"] == pytest.approx([8.0, 8.1, 9.0, 9.1])
    assert bank_d["x_t1"] == pytest.approx([0.6, 0.7, -0.1, 0.0])
    assert bank_d["hatc_t1"] == pytest.approx([-4.9, -4.8, -5.9, -5.8])


# ---------------------------------------------------------------------------
# bank structure: B / C / E + per-checkpoint transition rebuilding
# ---------------------------------------------------------------------------


def test_bank_b_counterfactual_eta_views():
    firm_frame, macro_frame = _synthetic_panel()
    bank_a = build_bank_a(firm_frame, macro_frame, seed=7, config_hash=CFG_HASH)
    bank_b = build_bank_b(bank_a, seed=8, config_hash=CFG_HASH)

    assert bank_b["parent_states"].shape == (8, 7)
    assert bank_b["parent_states"][:4, 2].tolist() == pytest.approx([0.0] * 4)
    assert bank_b["parent_states"][4:, 2].tolist() == pytest.approx([1.0] * 4)
    assert bank_b["row_weights"].tolist() == pytest.approx([1.0 / 8.0] * 8)
    assert bank_b["eta_share"] == {"eta0": 0.5, "eta1": 0.5}
    assert bank_b["pairing_keys"]["eta_view"] == ["eta0"] * 4 + ["eta1"] * 4
    assert bank_b["metadata"]["counterfactual-eta-view"] is True
    # non-eta state columns are identical across the two views
    for column in (0, 1, 3, 4, 5, 6):
        assert bank_b["parent_states"][:4, column].tolist() == bank_b["parent_states"][4:, column].tolist()


def test_bank_c_anchors_come_from_bank_d_joint_rows():
    firm_frame, macro_frame = _synthetic_panel()
    bank_d = build_bank_d(macro_frame, seed=7, config_hash=CFG_HASH)
    anchors = derive_macro_anchors(bank_d)
    assert set(anchors.keys()) == {"Recession", "Normal", "Expansion"}

    for anchor in anchors.values():
        row = int(anchor["source_row"])
        # joint sample: (x, hatcf, lnkf, hatc, lnk) all come from one Bank D row
        assert anchor["x"] == pytest.approx(bank_d["x_t"][row])
        assert anchor["hatcf"] == pytest.approx(bank_d["hatcf_t"][row])
        assert anchor["lnkf"] == pytest.approx(bank_d["lnkf_t"][row])
        assert anchor["hatc_cal"] == pytest.approx(bank_d["hatc_t"][row])
        assert anchor["lnk_cal"] == pytest.approx(bank_d["lnk_t"][row])

    x_by_label = {label: anchor["x"] for label, anchor in anchors.items()}
    assert x_by_label["Recession"] <= x_by_label["Normal"] <= x_by_label["Expansion"]

    bank_c = build_bank_c(seed=9, macro_anchors=anchors, config_hash=CFG_HASH)
    assert bank_c["tags"] == ["out-of-distribution-stress"]
    assert bank_c["n_rows"] == 5 * 5 * 3 * 2 * 3
    first = bank_c["parent_states"][0]
    assert first.tolist() == pytest.approx(
        [0.0, -2.5, 0.0, 0.10, anchors["Recession"]["x"], anchors["Recession"]["hatcf"], anchors["Recession"]["lnkf"]]
    )
    assert float(bank_c["hatc_cal"][0]) == pytest.approx(anchors["Recession"]["hatc_cal"])
    assert set(bank_c["parent_states"][:, 2].tolist()) == {0.0, 1.0}
    assert sorted({round(float(value), 5) for value in bank_c["parent_states"][:, 3].tolist()}) == [0.10, 0.25, 0.45]
    assert float(bank_c["parent_states"][:, 1].min()) == -2.5
    assert float(bank_c["parent_states"][:, 1].max()) == 2.5


def test_bank_e_variants_keep_capital_totals_comparable():
    bank_e = build_bank_e(
        seed=11,
        n_firms=8,
        n_paths=2,
        horizon=5,
        economic_config=AnalysisEconomicConfig.from_current_config(),
        config_hash=CFG_HASH,
    )
    variants = bank_e["variants"]
    assert set(variants.keys()) == {"Low", "Baseline", "High"}
    totals = [variants[name]["K0_total"] for name in ("Low", "Baseline", "High")]
    assert totals[0] == pytest.approx(totals[1])
    assert totals[1] == pytest.approx(totals[2])
    assert torch.all(variants["Low"]["b0"] <= variants["Baseline"]["b0"] + 1e-6)
    assert torch.all(variants["Baseline"]["b0"] <= variants["High"]["b0"] + 1e-6)
    assert torch.equal(variants["Low"]["K0"], variants["High"]["K0"])
    assert bank_e["shock_spec"]["rng_mode"] == "stable_firm_identity"
    assert bank_e["shock_spec"]["key_fields"] == [
        "seed",
        "path",
        "economic_time",
        "branch",
        "identity",
        "shock_type",
    ]
    assert bank_e["metadata"]["initial_hstar"]


def test_build_transition_for_checkpoint_rebuilds_children_and_gates_b():
    firm_frame, macro_frame = _synthetic_panel()
    bank_a = build_bank_a(firm_frame, macro_frame, seed=7, config_hash=CFG_HASH)
    sdf_model = build_models(torch.device("cpu"))["sdf_fc1"]
    hyperparams = HyperParams()
    economic_config = AnalysisEconomicConfig.from_current_config()

    transition = build_transition_for_checkpoint(bank_a, sdf_model, hyperparams, economic_config)
    repeat = build_transition_for_checkpoint(bank_a, sdf_model, hyperparams, economic_config)

    n_parent = bank_a["parent_states"].shape[0]
    n_child = bank_a["shock_bank"]["eps_x"].shape[1]
    assert transition.children.shape == (n_parent, 2 * n_child, 7)
    assert transition.m_raw.shape == (n_parent, 2 * n_child, 1)
    assert transition.m_used.shape == (n_parent, 2 * n_child, 1)
    assert transition.branch_weights.shape == (n_parent, 2 * n_child)
    assert torch.equal(transition.children, repeat.children)
    assert bool(torch.isfinite(transition.m_raw).all())
    assert bool(torch.isfinite(transition.m_used).all())
    assert torch.equal(transition.m_raw, repeat.m_raw)
    assert torch.equal(transition.m_used, repeat.m_used)
    assert transition.metadata["shock_bank_sha256"] == bank_a["shock_bank_sha256"]
    assert transition.metadata["eta_integration_mode"] == "exact"

    # exact-eta expansion enumerates eta=0/1 pairs over the parent eta gate
    eta_column = transition.children[:, :, 2]
    assert set(eta_column.reshape(-1).tolist()) == {0.0, 1.0}

    bp = torch.full((n_parent, 1), 0.77)
    parent_b = bank_a["parent_states"][:, 0]
    parent_eta = bank_a["parent_states"][:, 2]
    expected = parent_eta * 0.77 + (1.0 - parent_eta) * parent_b
    gated = transition.child_b_with_policy(bp)
    assert gated.shape == (n_parent, 2 * n_child, 1)
    assert gated[:, 0, 0].tolist() == pytest.approx(expected.tolist())

    forced = build_transition_for_checkpoint(
        bank_a, sdf_model, hyperparams, economic_config, eta_override=0.0
    )
    assert forced.child_b_with_policy(bp)[:, 0, 0].tolist() == pytest.approx(parent_b.tolist())
    refinanced = build_transition_for_checkpoint(
        bank_a, sdf_model, hyperparams, economic_config, eta_override=1.0
    )
    assert refinanced.child_b_with_policy(bp)[:, 0, 0].tolist() == pytest.approx([0.77] * n_parent)
    assert refinanced.metadata["eta_override"] == 1.0
