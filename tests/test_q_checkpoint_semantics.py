"""Checkpoint / model-spec semantic metadata for direct-Q.

A legacy ``b * q_unit`` state_dict is tensor-shape compatible with direct-Q, so
``load_state_dict(strict=True)`` alone cannot detect the mismatch. These tests lock
the guard on the raw ``build_models(..., ckpt_dir=...)`` path and the metadata that
``save_models`` now writes.
"""

import json
from pathlib import Path

import pytest
import torch

from config import Config
from config.hyperparams import HyperParams
from experiments.run_utils import (
    build_models,
    load_policy_value_model_spec,
    save_models,
)
from models.policy_value import PolicyValueModel


def _policy_value_model(q_parameterization: str) -> PolicyValueModel:
    """Construct a model with the right semantics.

    ``q_parameterization`` 会改变 ``q_head`` 的输出激活（direct -> None，
    b_times_unit -> softplus），因此必须构造时指定，不能事后改属性。
    """
    return PolicyValueModel(q_parameterization=q_parameterization)


def _device() -> torch.device:
    return torch.device("cpu")


def _raw_checkpoint_dir(tmp_path: Path) -> Path:
    """Legacy layout: bare state_dicts, no ``metadata/``."""
    models = build_models(_device())
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(models["sdf_fc1"].state_dict(), ckpt_dir / "ep0_sdf_fc1.pt")
    torch.save(models["policy_value"].state_dict(), ckpt_dir / "ep0_policy_value.pt")
    torch.save(models["fc2"].state_dict(), ckpt_dir / "ep0_fc2.pt")
    return ckpt_dir


def _write_spec(tmp_path: Path, *, q_parameterization: str) -> None:
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    spec = _policy_value_model(q_parameterization).model_spec()
    (meta_dir / "policy_value_model_spec.json").write_text(
        json.dumps({"policy_value_model_spec": spec}), encoding="utf-8"
    )


# ---------------------------------------------------------------- TEST F

def test_F_raw_checkpoint_is_rejected_when_direct_q_expected(tmp_path, monkeypatch):
    ckpt_dir = _raw_checkpoint_dir(tmp_path)
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "direct")
    with pytest.raises(ValueError, match="without policy_value_model_spec"):
        build_models(_device(), ckpt_dir=ckpt_dir, ckpt_prefix="ep0", strict=True)


def test_F2_allow_unsafe_alone_does_not_decide_semantics(tmp_path, monkeypatch):
    ckpt_dir = _raw_checkpoint_dir(tmp_path)
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "direct")
    with pytest.raises(ValueError, match="raw_q_parameterization"):
        build_models(
            _device(),
            ckpt_dir=ckpt_dir,
            ckpt_prefix="ep0",
            strict=True,
            allow_unsafe_raw_checkpoint=True,
        )


def test_F3_explicit_mode_is_required_even_for_legacy_target(tmp_path, monkeypatch):
    ckpt_dir = _raw_checkpoint_dir(tmp_path)
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "b_times_unit")
    with pytest.raises(ValueError, match="raw_q_parameterization"):
        build_models(_device(), ckpt_dir=ckpt_dir, ckpt_prefix="ep0", strict=True)
    models = build_models(
        _device(),
        ckpt_dir=ckpt_dir,
        ckpt_prefix="ep0",
        strict=True,
        raw_q_parameterization="b_times_unit",
    )
    assert models["policy_value"].q_parameterization == "b_times_unit"


def test_F6_invalid_raw_q_parameterization_is_rejected(tmp_path, monkeypatch):
    ckpt_dir = _raw_checkpoint_dir(tmp_path)
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "direct")
    with pytest.raises(ValueError, match="raw_q_parameterization must be one of"):
        build_models(
            _device(),
            ckpt_dir=ckpt_dir,
            ckpt_prefix="ep0",
            strict=True,
            raw_q_parameterization="q_unit",
        )


def test_F4_spec_semantics_win_over_current_config(tmp_path, monkeypatch):
    """evaluation 默认相信 checkpoint spec，不再拿 Config.Q_PARAMETERIZATION 去卡它。"""
    ckpt_dir = _raw_checkpoint_dir(tmp_path)
    _write_spec(tmp_path, q_parameterization="b_times_unit")
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "direct")
    models = build_models(_device(), ckpt_dir=ckpt_dir, ckpt_prefix="ep0", strict=True)
    assert models["policy_value"].q_parameterization == "b_times_unit"


def test_F5_matching_spec_loads_without_bypass(tmp_path, monkeypatch):
    ckpt_dir = _raw_checkpoint_dir(tmp_path)
    _write_spec(tmp_path, q_parameterization="direct")
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "direct")
    models = build_models(_device(), ckpt_dir=ckpt_dir, ckpt_prefix="ep0", strict=True)
    assert models["policy_value"].q_parameterization == "direct"


# ---------------------------------------------------------------- TEST G

def test_G_save_models_writes_semantic_metadata(tmp_path):
    models = build_models(_device())
    paths = save_models(models, 3, tmp_path, hyperparams=HyperParams())
    for key in (
        "hyperparams",
        "config_snapshot",
        "policy_value_model_spec",
        "combined_checkpoint",
    ):
        assert Path(paths[key]).exists(), key
    assert paths["policy_value_model_spec"].endswith(
        "metadata/policy_value_model_spec.json"
    )
    assert paths["combined_checkpoint"].endswith("checkpoints/ep3_combined.pt")
    spec = load_policy_value_model_spec(tmp_path / "checkpoints", "ep3")
    assert spec == models["policy_value"].model_spec()
    assert spec["q_parameterization"] == "direct"


def test_G2_combined_checkpoint_carries_semantics_and_firm_target(tmp_path):
    models = build_models(_device())
    save_models(
        models,
        1,
        tmp_path,
        hyperparams=HyperParams(),
        extra_models={"firm_target": models["policy_value"]},
    )
    payload = torch.load(tmp_path / "checkpoints" / "ep1_combined.pt", map_location="cpu")
    assert payload["policy_value_model_spec"]["q_parameterization"] == "direct"
    assert payload["hyperparams"]["q_parameterization"] == "direct"
    assert "config_snapshot" in payload
    assert set(payload["models"]) == {"sdf_fc1", "policy_value", "fc2", "firm_target"}


def test_G3_saved_metadata_round_trips_through_build_models(tmp_path, monkeypatch):
    models = build_models(_device())
    save_models(models, 0, tmp_path, hyperparams=HyperParams())
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "direct")
    reloaded = build_models(
        _device(), ckpt_dir=tmp_path / "checkpoints", ckpt_prefix="ep0", strict=True
    )
    for name in ("policy_value", "sdf_fc1", "fc2"):
        for key, value in models[name].state_dict().items():
            torch.testing.assert_close(reloaded[name].state_dict()[key], value)


def test_G4_hyperparams_json_is_readable_and_records_q_semantics(tmp_path):
    models = build_models(_device())
    hp = HyperParams()
    save_models(models, 2, tmp_path, hyperparams=hp)
    payload = json.loads(
        (tmp_path / "metadata" / "hyperparams.json").read_text(encoding="utf-8")
    )
    assert payload["q_parameterization"] == "direct"
    assert "q_bootstrap_epochs" in payload
    assert "q_require_survival_phase" in payload


# ---------------------------------------------------------------- TEST H

def test_H_shape_weights_default_to_zero_and_are_overridable():
    hp = HyperParams()
    assert hp.q_shape_weight_z == 0.0
    assert hp.q_shape_weight_b_low == 0.0
    assert hp.q_shape_weight_b_high == 0.0
    hp.q_shape_weight_z = 1.0
    hp.q_shape_weight_b_low = 1.0
    hp.q_shape_weight_b_high = 1.0
    assert hp.q_shape_weight_z == 1.0
    assert hp.q_shape_weight_b_low == 1.0
    assert hp.q_shape_weight_b_high == 1.0


def test_H2_zero_shape_weights_remove_the_penalty_from_the_q_objective():
    from training.episode import Episode

    hp = HyperParams()
    hp.policy_lr = 1e-3
    hp.q_zero_boundary_epochs = 1
    hp.q_default_pretrain_epochs = 0
    hp.q_survival_aio_epochs = 1
    hp.q_mixed_polish_epochs = 0
    device = _device()
    online = build_models(device)["policy_value"]
    target = build_models(device)["policy_value"]
    target.load_state_dict(online.state_dict())
    episode = Episode(
        models={"policy_value": online},
        optimizers={"policy_value": torch.optim.AdamW(online.parameters(), lr=hp.policy_lr)},
        config=Config,
        hyperparams=hp,
        device=device,
        firm_target=target,
    )
    parent = torch.tensor(
        [
            [0.10, 0.20, 1.00, 0.20, 0.10, -2.00, 4.00],
            [0.30, 0.10, 0.00, 0.40, 0.00, -1.80, 4.20],
        ],
        dtype=torch.float32,
    )
    child0 = parent.clone()
    child1 = parent.clone()
    child0[:, 1:2] += 0.05
    child1[:, 1:2] -= 0.03
    child0[:, 2:3] = torch.tensor([[1.0], [0.0]])
    child1[:, 2:3] = torch.tensor([[0.0], [1.0]])
    batch = {
        "parent": torch.cat([parent, torch.ones(2, 1)], dim=1),
        "children": [
            torch.cat([child0, torch.full((2, 1), 0.98)], dim=1),
            torch.cat([child1, torch.full((2, 1), 1.02)], dim=1),
        ],
    }
    # 形状先验默认关闭：diagnostics key 保留但恒为 0。
    episode.hyperparams.q_shape_weight_z = 0.0
    episode.hyperparams.q_shape_weight_b_low = 0.0
    episode.hyperparams.q_shape_weight_b_high = 0.0
    episode._compute_q_survival_bellman_loss(batch, create_graph=False, q_target_model=target)
    zero_terms = dict(episode._latest_q_terms)
    assert zero_terms["q_shape_z"] == 0.0
    assert zero_terms["q_shape_b_low"] == 0.0
    assert zero_terms["q_shape_b_high"] == 0.0
    # 打开形状先验后必须重新真的计算 shape derivative（不再是 0）。
    episode.hyperparams.q_shape_weight_z = 5.0
    episode._compute_q_survival_bellman_loss(batch, create_graph=False, q_target_model=target)
    weighted_terms = dict(episode._latest_q_terms)
    assert weighted_terms["q_shape_z"] >= 0.0
    assert weighted_terms["q_physics"] >= zero_terms["q_physics"] - 1e-9


# ------------------------------------------------- TEST 1: scaled-value metadata

def test_T1_scaled_value_metadata_matches_hyperparams(tmp_path):
    from analysis.checkpoint_loader import load_analysis_checkpoint

    models = build_models(_device())
    hp = HyperParams()
    hp.pv_value_scale_mode = "exp_xz"
    hp.pv_value_scale_log_max = 17.0
    hp.pv_bellman_normalize_by_value_scale = True
    save_models(models, 4, tmp_path, hyperparams=hp)

    payload = torch.load(tmp_path / "checkpoints" / "ep4_combined.pt", map_location="cpu")
    assert payload["hyperparams"]["pv_value_scale_mode"] == "exp_xz"
    assert payload["value_parameterization"]["mode"] == "exp_xz"
    assert payload["value_parameterization"]["log_max"] == 17.0
    assert payload["value_parameterization"]["bellman_normalization"] is True
    # 两边完全一致：不再出现 Config 推断出的 mode=none / log_max=20。
    assert (
        payload["value_parameterization"]["mode"]
        == str(payload["hyperparams"]["pv_value_scale_mode"]).lower()
    )
    assert (
        payload["value_parameterization"]["log_max"]
        == payload["hyperparams"]["pv_value_scale_log_max"]
    )
    assert (
        payload["value_parameterization"]["bellman_normalization"]
        == payload["hyperparams"]["pv_bellman_normalize_by_value_scale"]
    )
    assert payload["value_parameterization"]["scale_formula"] == "1+exp(clamp(x+z,max=17))"

    # strict loader round-trip：metadata 与 hyperparams 一致时不得报 mismatch。
    result = load_analysis_checkpoint(
        tmp_path / "checkpoints" / "ep4_combined.pt",
        hyperparams_json=tmp_path / "metadata" / "hyperparams.json",
        config_json=tmp_path / "metadata" / "config_snapshot.json",
        device="cpu",
    )
    loaded_meta = result.metadata["value_parameterization"]
    assert loaded_meta["checkpoint"]["mode"] == "exp_xz"
    assert loaded_meta["checkpoint"]["log_max"] == 17.0
    assert loaded_meta["current"]["mode"] == "exp_xz"
    assert loaded_meta["current"]["log_max"] == 17.0


def test_T1b_config_source_would_have_disagreed(tmp_path, monkeypatch):
    """回归锁定：Config 上的 PV_VALUE_SCALE_* 与 HyperParams 不一致时，metadata 取 HyperParams。"""
    monkeypatch.setattr(Config, "PV_VALUE_SCALE_MODE", "none")
    monkeypatch.setattr(Config, "PV_VALUE_SCALE_LOG_MAX", 20.0)
    monkeypatch.setattr(Config, "PV_BELLMAN_NORMALIZE_BY_VALUE_SCALE", False)
    models = build_models(_device())
    hp = HyperParams()
    hp.pv_value_scale_mode = "exp_xz"
    hp.pv_value_scale_log_max = 17.0
    hp.pv_bellman_normalize_by_value_scale = True
    save_models(models, 5, tmp_path, hyperparams=hp)
    payload = torch.load(tmp_path / "checkpoints" / "ep5_combined.pt", map_location="cpu")
    assert payload["value_parameterization"]["mode"] == "exp_xz"
    assert payload["value_parameterization"]["log_max"] == 17.0
    assert payload["value_parameterization"]["bellman_normalization"] is True


# --------------------------------------- TEST 2/3/4: raw checkpoint Q semantics

def _legacy_raw_checkpoint(tmp_path: Path) -> Path:
    """Bare state_dict from a legacy ``Q = b * q_unit`` model."""
    legacy = _policy_value_model("b_times_unit")
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(legacy.state_dict(), ckpt_dir / "ep0_policy_value.pt")
    return ckpt_dir


def _sample_state() -> torch.Tensor:
    state = torch.zeros((4, 7))
    state[:, 0] = torch.tensor([0.0, 0.25, 0.60, 1.20])
    state[:, 1] = torch.tensor([-1.0, -0.2, 0.4, 1.5])
    state[:, 2] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    state[:, 3] = 0.3
    state[:, 4] = 0.05
    state[:, 5] = -1.8
    state[:, 6] = 4.2
    return state


def test_T2_legacy_raw_round_trips_with_b_times_unit_semantics(tmp_path, monkeypatch):
    ckpt_dir = _legacy_raw_checkpoint(tmp_path)
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "b_times_unit")
    loaded = build_models(
        _device(),
        ckpt_dir=ckpt_dir,
        ckpt_prefix="ep0",
        strict=True,
        raw_q_parameterization="b_times_unit",
    )["policy_value"]
    assert loaded.q_parameterization == "b_times_unit"

    legacy = _policy_value_model("b_times_unit")
    legacy.load_state_dict(torch.load(ckpt_dir / "ep0_policy_value.pt", map_location="cpu"))
    state = _sample_state()
    with torch.no_grad():
        q_loaded = loaded._q_output(state)
        q_legacy = legacy._q_output(state)
        base_state, _, b_legacy = legacy._split_state(state)
        q_unit = legacy.q_head(legacy.q_encoder(base_state))
    torch.testing.assert_close(q_loaded, q_legacy)
    # Q = b * q_unit，而不是 Q = q_unit。
    torch.testing.assert_close(q_loaded, torch.clamp(b_legacy, min=0.0) * q_unit)
    assert not torch.allclose(q_loaded, q_unit)


def test_T3_legacy_raw_cannot_be_silently_loaded_as_direct(tmp_path, monkeypatch):
    ckpt_dir = _legacy_raw_checkpoint(tmp_path)
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "direct")
    with pytest.raises(ValueError, match="raw_q_parameterization"):
        build_models(_device(), ckpt_dir=ckpt_dir, ckpt_prefix="ep0", strict=True)
    # 显式声明成 direct 也是错的语义，但至少不是 silent；此处只锁定"不静默"。
    with pytest.raises(ValueError, match="raw_q_parameterization"):
        build_models(
            _device(),
            ckpt_dir=ckpt_dir,
            ckpt_prefix="ep0",
            strict=True,
            allow_unsafe_raw_checkpoint=True,
        )


def test_T4_direct_raw_round_trips_with_explicit_direct_semantics(tmp_path, monkeypatch):
    direct = _policy_value_model("direct")
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(direct.state_dict(), ckpt_dir / "ep0_policy_value.pt")
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "direct")
    loaded = build_models(
        _device(),
        ckpt_dir=ckpt_dir,
        ckpt_prefix="ep0",
        strict=True,
        raw_q_parameterization="direct",
    )["policy_value"]
    assert loaded.q_parameterization == "direct"
    state = _sample_state()
    with torch.no_grad():
        q_loaded = loaded._q_output(state)
        q_direct = direct._q_output(state)
        base_state, _, b_direct = direct._split_state(state)
        q_unit = direct.q_head(direct.q_encoder(base_state))
        scaled = torch.clamp(b_direct, min=0.0) * q_unit
    torch.testing.assert_close(q_loaded, q_direct)
    # direct-Q 不再乘 b。
    assert not torch.allclose(q_loaded, scaled)


# ============================================ TEST 1-5: evaluation vs resume

def _metadata_checkpoint(tmp_path: Path, *, q_parameterization: str):
    """Metadata checkpoint whose spec declares ``q_parameterization``."""
    model = _policy_value_model(q_parameterization)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "ep0_policy_value.pt")
    spec = model.model_spec()
    spec["q_parameterization"] = q_parameterization
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "policy_value_model_spec.json").write_text(
        json.dumps({"policy_value_model_spec": spec}), encoding="utf-8"
    )
    return ckpt_dir, model


def test_T1_metadata_legacy_checkpoint_evaluates_under_direct_config(tmp_path, monkeypatch):
    ckpt_dir, legacy = _metadata_checkpoint(tmp_path, q_parameterization="b_times_unit")
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "direct")
    loaded = build_models(
        _device(), ckpt_dir=ckpt_dir, ckpt_prefix="ep0", strict=True
    )["policy_value"]
    assert loaded.q_parameterization == "b_times_unit"
    state = _sample_state()
    with torch.no_grad():
        q_loaded = loaded._q_output(state)
        q_legacy = legacy._q_output(state)
        base_state, _, b_legacy = legacy._split_state(state)
        q_unit = legacy.q_head(legacy.q_encoder(base_state))
    torch.testing.assert_close(q_loaded, q_legacy)
    # 没有被 reinterpret 成 direct-Q：仍然是 Q = b * q_unit。
    torch.testing.assert_close(q_loaded, torch.clamp(b_legacy, min=0.0) * q_unit)
    assert not torch.allclose(q_loaded, q_unit)


def test_T2_metadata_legacy_checkpoint_resume_to_direct_is_rejected(tmp_path, monkeypatch):
    ckpt_dir, _ = _metadata_checkpoint(tmp_path, q_parameterization="b_times_unit")
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "direct")
    with pytest.raises(ValueError) as excinfo:
        build_models(
            _device(),
            ckpt_dir=ckpt_dir,
            ckpt_prefix="ep0",
            strict=True,
            expected_q_parameterization="direct",
        )
    message = str(excinfo.value)
    assert "b_times_unit" in message
    assert "expected_q_parameterization='direct'" in message


def test_T3_matching_expected_semantics_passes(tmp_path, monkeypatch):
    direct_dir, _ = _metadata_checkpoint(tmp_path / "d", q_parameterization="direct")
    legacy_dir, _ = _metadata_checkpoint(tmp_path / "l", q_parameterization="b_times_unit")
    # Config 与两者都不同，不应影响任何一边。
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "b_times_unit")
    direct_loaded = build_models(
        _device(),
        ckpt_dir=direct_dir,
        ckpt_prefix="ep0",
        strict=True,
        expected_q_parameterization="direct",
    )["policy_value"]
    legacy_loaded = build_models(
        _device(),
        ckpt_dir=legacy_dir,
        ckpt_prefix="ep0",
        strict=True,
        expected_q_parameterization="b_times_unit",
    )["policy_value"]
    assert direct_loaded.q_parameterization == "direct"
    assert legacy_loaded.q_parameterization == "b_times_unit"


def test_T4_raw_checkpoint_rules_unchanged(tmp_path, monkeypatch):
    ckpt_dir = _raw_checkpoint_dir(tmp_path)
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "b_times_unit")
    # metadata 缺失 + raw_q_parameterization=None -> reject（Config 不参与猜测）
    with pytest.raises(ValueError, match="raw_q_parameterization"):
        build_models(_device(), ckpt_dir=ckpt_dir, ckpt_prefix="ep0", strict=True)
    direct_loaded = build_models(
        _device(), ckpt_dir=ckpt_dir, ckpt_prefix="ep0", strict=True,
        raw_q_parameterization="direct",
    )["policy_value"]
    legacy_loaded = build_models(
        _device(), ckpt_dir=ckpt_dir, ckpt_prefix="ep0", strict=True,
        raw_q_parameterization="b_times_unit",
    )["policy_value"]
    assert direct_loaded.q_parameterization == "direct"
    assert legacy_loaded.q_parameterization == "b_times_unit"
    # raw semantic != expected semantic -> reject
    with pytest.raises(ValueError, match="expected_q_parameterization"):
        build_models(
            _device(), ckpt_dir=ckpt_dir, ckpt_prefix="ep0", strict=True,
            raw_q_parameterization="b_times_unit",
            expected_q_parameterization="direct",
        )
    matched = build_models(
        _device(), ckpt_dir=ckpt_dir, ckpt_prefix="ep0", strict=True,
        raw_q_parameterization="b_times_unit",
        expected_q_parameterization="b_times_unit",
    )["policy_value"]
    assert matched.q_parameterization == "b_times_unit"


def test_T5_checkpoint_spec_beats_current_config(tmp_path, monkeypatch):
    """本轮最关键的 regression：Config=direct 时仍能忠实恢复 legacy checkpoint。"""
    ckpt_dir, legacy = _metadata_checkpoint(tmp_path, q_parameterization="b_times_unit")
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "direct")
    loaded = build_models(
        _device(), ckpt_dir=ckpt_dir, ckpt_prefix="ep0", strict=True
    )["policy_value"]
    assert loaded.q_parameterization == "b_times_unit"
    assert loaded.q_parameterization == legacy.q_parameterization
    # Config 只决定没有 checkpoint 时的 fresh model。
    assert build_models(_device())["policy_value"].q_parameterization == "direct"
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "b_times_unit")
    assert build_models(_device())["policy_value"].q_parameterization == "b_times_unit"


def test_T5b_invalid_expected_q_parameterization_is_rejected(tmp_path, monkeypatch):
    ckpt_dir, _ = _metadata_checkpoint(tmp_path, q_parameterization="direct")
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "direct")
    with pytest.raises(ValueError, match="expected_q_parameterization must be one of"):
        build_models(
            _device(), ckpt_dir=ckpt_dir, ckpt_prefix="ep0", strict=True,
            expected_q_parameterization="q_unit",
        )
