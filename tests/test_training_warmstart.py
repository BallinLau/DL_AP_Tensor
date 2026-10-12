from copy import deepcopy
from pathlib import Path

import pytest
import torch

from config import Config
from config.hyperparams import HyperParams
from experiments.run_multi_episode_job import resolve_start_episode
from experiments.run_utils import (
    build_models,
    build_optimizers,
    load_training_warmstart,
    save_models,
)
from training.episode import Episode


@pytest.fixture(autouse=True)
def _pin_direct_q_model_semantics(monkeypatch):
    # Other tests intentionally mutate this process-global Config field. Keep
    # this checkpoint fixture independent of test execution order.
    monkeypatch.setattr(Config, "Q_PARAMETERIZATION", "direct")


def _assert_same_state(left, right):
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key])


def _write_checkpoint(tmp_path: Path):
    device = torch.device("cpu")
    models = build_models(device)
    firm_target = deepcopy(models["policy_value"])
    with torch.no_grad():
        first_param = next(firm_target.parameters())
        first_param.add_(0.25)

    hp = HyperParams()
    hp.q_parameterization = "direct"
    hp.pv_value_scale_mode = "exp_xz"
    hp.pv_value_scale_log_max = 20.0
    hp.pv_bellman_normalize_by_value_scale = True
    paths = save_models(
        models,
        9,
        tmp_path,
        hyperparams=hp,
        extra_models={"firm_target": firm_target},
    )
    return models, firm_target, Path(paths["combined_checkpoint"]), hp


def test_continuation_warmstart_restores_all_models_and_firm_target(tmp_path):
    original, original_target, checkpoint, hp = _write_checkpoint(tmp_path)
    loaded, loaded_target, provenance = load_training_warmstart(
        checkpoint,
        torch.device("cpu"),
        expected_episode=9,
        expected_q_parameterization="direct",
        expected_value_scale_mode="exp_xz",
        expected_value_scale_log_max=20.0,
        expected_bellman_normalization=True,
    )

    for name in ("policy_value", "sdf_fc1", "fc2"):
        _assert_same_state(original[name].state_dict(), loaded[name].state_dict())
    _assert_same_state(original_target.state_dict(), loaded_target.state_dict())
    assert any(
        not torch.equal(online, target)
        for online, target in zip(
            loaded["policy_value"].state_dict().values(),
            loaded_target.state_dict().values(),
        )
    )
    assert all(not param.requires_grad for param in loaded_target.parameters())
    assert provenance["source_episode"] == 9
    assert provenance["optimizer_state_restored"] is False
    assert provenance["rng_state_restored"] is False

    optimizers = build_optimizers(loaded, hp)
    episode = Episode(
        models=loaded,
        optimizers=optimizers,
        config=Config,
        hyperparams=hp,
        device=torch.device("cpu"),
        episode_id=10,
        firm_target=loaded_target,
        q_checkpoint_loaded=True,
    )
    assert episode._q_bootstrap_enabled() is False
    _assert_same_state(original_target.state_dict(), episode.firm_target.state_dict())


def test_warmstart_rejects_semantic_mismatch(tmp_path):
    _, _, checkpoint, _ = _write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="q_parameterization mismatch"):
        load_training_warmstart(
            checkpoint,
            torch.device("cpu"),
            expected_q_parameterization="b_times_unit",
            expected_value_scale_mode="exp_xz",
            expected_value_scale_log_max=20.0,
            expected_bellman_normalization=True,
        )
    with pytest.raises(ValueError, match="Bellman normalization mismatch"):
        load_training_warmstart(
            checkpoint,
            torch.device("cpu"),
            expected_q_parameterization="direct",
            expected_value_scale_mode="exp_xz",
            expected_value_scale_log_max=20.0,
            expected_bellman_normalization=False,
        )


def test_start_episode_is_next_logical_episode_only():
    provenance = {"source_episode": 9}
    assert resolve_start_episode(
        requested_start_episode=None,
        warmstart_provenance=provenance,
    ) == 10
    assert resolve_start_episode(
        requested_start_episode=10,
        warmstart_provenance=provenance,
    ) == 10
    with pytest.raises(ValueError, match="must begin at the next logical episode"):
        resolve_start_episode(
            requested_start_episode=0,
            warmstart_provenance=provenance,
        )
    with pytest.raises(ValueError, match="must be 0"):
        resolve_start_episode(
            requested_start_episode=10,
            warmstart_provenance=None,
        )
