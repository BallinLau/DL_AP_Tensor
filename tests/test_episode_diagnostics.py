import hashlib
import json

import numpy as np
import pandas as pd
import torch

from evaluation.episode_diagnostics import save_episode_diagnostics
from models.policy_value import PolicyValueModel


def _model() -> PolicyValueModel:
    return PolicyValueModel(
        q_parameterization="hybrid_regime",
        share_hidden_dims=[4],
        share_output_dim=4,
        q_head_dims=[4],
        p0_head_dims=[4],
        pi_head_dims=[4],
        bp0_head_dims=[4],
        bpi_head_dims=[4],
        barz_hidden_dims=[4],
        bari_hidden_dims=[4],
        i_grid_size=2,
    )


def _hash(model) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _firm_df(shift=0.0):
    return pd.DataFrame({
        "branch": [-1, -1, 0],
        "x": [0.1 + shift, 0.3 + shift, 0.2],
        "i": [0.2, 0.4, 0.3],
        "Hatcf": [-2.0, -1.8, -1.9],
        "LnKF": [4.0, 4.2, 4.1],
        "P": [1.0, 0.0, 0.5],
        "Q": [0.2, 0.3, 0.25],
        "b": [0.1, 0.5, 0.3],
        "bp": [0.2, 0.6, 0.4],
        "z": [-1.0, 1.0, 0.0],
        "M": [0.9, 1.0, 1.1],
        "Bar_i": [0.1, 0.2, 0.3],
    })


def _summary():
    history = {
        "phase": "survival",
        "epoch": 1,
        "score_primary": 0.2,
        "score_raw_abs": 2.0,
        "score_normalized_abs": 0.2,
        "q_unit_mean": 1.0,
        "q_unit_p95": 1.2,
        "q_unit_max": 1.3,
        "q_claim_value_mean": 0.4,
        "q_claim_value_p95": 0.7,
        "q_claim_value_max": 0.8,
        "q_recursion_gain_mean": 0.8,
        "q_recursion_gain_p95": 1.1,
        "q_recursion_gain_gt1_share": 0.2,
        "lr": 1e-4,
        "target_hash": "target",
        "online_q_hash": "online",
        "is_best": True,
        "restored": False,
    }
    q_phase = {
        "phase": "survival",
        "status": "accepted_improved",
        "metrics": history,
        "validation_start": {"score_primary": 0.3},
        "validation_best": {"score_primary": 0.2},
        "validation_after_restore": {"score_primary": 0.2},
    }
    return {
        "policy_value": {
            "metadata": {
                "policy_value_evaluation_stage": {"status": "accepted"},
                "q_regime_training_stage": {
                    "status": "accepted_improved",
                    "q_no_improvement_streak": 0,
                    "q_online_hash_stage_start": "start",
                    "phases": [q_phase],
                    "q_validation_history": [history],
                },
                "bp_distillation_stage": {"status": "accepted"},
            }
        }
    }


def test_episode_diagnostics_artifacts_and_fixed_canonical_reference(tmp_path):
    model = _model()
    before = _hash(model)
    reference = save_episode_diagnostics(
        run_root=tmp_path,
        episode=0,
        model=model,
        firm_df=_firm_df(),
        module_summary=_summary(),
    )
    second = save_episode_diagnostics(
        run_root=tmp_path,
        episode=1,
        model=model,
        firm_df=_firm_df(shift=2.0),
        module_summary=_summary(),
        canonical_reference=reference,
    )
    assert second == reference
    assert _hash(model) == before

    expected = {
        "00_episode_start.json",
        "10_post_p.json",
        "20_post_q_survival.json",
        "30_post_q_final.json",
        "40_post_bp.json",
        "episode_health.json",
        "q_validation_history.csv",
        "sentinel_states.csv",
        "surface_canonical.npz",
        "surface_ondist.npz",
    }
    for episode in (0, 1):
        directory = tmp_path / "episode_diagnostics" / f"ep_{episode:03d}"
        assert expected <= {path.name for path in directory.iterdir()}
        for name in expected:
            assert (directory / name).stat().st_size > 0
        json.loads((directory / "episode_health.json").read_text())

    with np.load(tmp_path / "episode_diagnostics/ep_000/surface_canonical.npz") as data0, np.load(
        tmp_path / "episode_diagnostics/ep_001/surface_canonical.npz"
    ) as data1:
        required = {
            "b_grid", "z_grid", "P0_eta0", "P0_eta1", "PI_eta0", "PI_eta1",
            "Phat_eta0", "Phat_eta1", "P_eta0", "P_eta1", "q_unit_eta0",
            "q_unit_eta1", "Q_claim_eta0", "Q_claim_eta1", "Q_effective_eta0",
            "Q_effective_eta1", "recovery_eta0", "recovery_eta1", "bp0_eta0",
            "bp0_eta1", "bpI_eta0", "bpI_eta1",
        }
        assert required <= set(data0.files)
        for key in ("reference_x", "reference_i", "reference_Hatcf", "reference_LnKF"):
            np.testing.assert_array_equal(data0[key], data1[key])

    with np.load(tmp_path / "episode_diagnostics/ep_000/surface_ondist.npz") as data0, np.load(
        tmp_path / "episode_diagnostics/ep_001/surface_ondist.npz"
    ) as data1:
        assert float(data0["reference_x"]) != float(data1["reference_x"])
