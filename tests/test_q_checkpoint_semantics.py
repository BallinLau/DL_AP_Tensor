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
    spec = build_models(_device())["policy_value"].model_spec()
    spec["q_parameterization"] = q_parameterization
    (meta_dir / "policy_value_model_spec.json").write_text(
        json.dumps({"policy_value_model_spec": spec}), encoding="utf-8"
    )


# ---------------------------------------------------------------- TEST F

def test_F_raw_checkpoint_is_rejected_when_direct_q_expected(tmp_path, monkeypatch):
    ckpt_dir = _raw_checkpoint_dir(tmp_path)
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "direct")
    with pytest.raises(ValueError, match="without policy_value_model_spec"):
        build_models(_device(), ckpt_dir=ckpt_dir, ckpt_prefix="ep0", strict=True)


def test_F2_raw_checkpoint_loads_only_with_explicit_opt_in(tmp_path, monkeypatch):
    ckpt_dir = _raw_checkpoint_dir(tmp_path)
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "direct")
    models = build_models(
        _device(),
        ckpt_dir=ckpt_dir,
        ckpt_prefix="ep0",
        strict=True,
        allow_unsafe_raw_checkpoint=True,
    )
    assert models["policy_value"].q_parameterization == "direct"


def test_F3_raw_checkpoint_is_unambiguous_for_legacy_expected_mode(tmp_path, monkeypatch):
    ckpt_dir = _raw_checkpoint_dir(tmp_path)
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "b_times_unit")
    models = build_models(_device(), ckpt_dir=ckpt_dir, ckpt_prefix="ep0", strict=True)
    assert models["policy_value"].q_parameterization == "b_times_unit"


def test_F4_spec_q_parameterization_mismatch_is_rejected(tmp_path, monkeypatch):
    ckpt_dir = _raw_checkpoint_dir(tmp_path)
    _write_spec(tmp_path, q_parameterization="b_times_unit")
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "direct")
    with pytest.raises(ValueError, match="q_parameterization mismatch"):
        build_models(_device(), ckpt_dir=ckpt_dir, ckpt_prefix="ep0", strict=True)


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
    # 形状先验默认关闭：即使 Q 对 b/z 的单调性被破坏，也不应有 shape penalty 贡献。
    episode.hyperparams.q_shape_weight_z = 0.0
    episode.hyperparams.q_shape_weight_b_low = 0.0
    episode.hyperparams.q_shape_weight_b_high = 0.0
    episode._compute_q_survival_bellman_loss(batch, create_graph=False, q_target_model=target)
    zero_terms = dict(episode._latest_q_terms)
    assert zero_terms["q_shape_z"] >= 0.0
    episode.hyperparams.q_shape_weight_z = 5.0
    episode._compute_q_survival_bellman_loss(batch, create_graph=False, q_target_model=target)
    weighted_terms = dict(episode._latest_q_terms)
    # 关闭时 physics 损失严格等于 loss + penalty_z（无 shape 项）。
    assert zero_terms["q_shape_z"] == pytest.approx(weighted_terms["q_shape_z"])
    assert zero_terms["q_physics"] <= weighted_terms["q_physics"] + 1e-9
