from pathlib import Path
import json
import random
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "tests"))

from config.hyperparams import HyperParams
from experiments.run_utils import build_models
from analysis.checkpoint_loader import load_analysis_checkpoint
from analysis.convergence_surface import (
    VectorizedBellmanSurfaceBackend,
    evaluate_checkpoint_convergence_surfaces,
    reduce_signed_surface,
)
from analysis.convergence_transition import (
    ChildExogenousBundle,
    ConvergenceShockBank,
    FirmStateIndex,
    MacroTransitionContext,
    build_child_exogenous_bundle,
)
from training.episode import Episode
from test_policy_value_stage_separation import _batch, _episode


def _write_combined_checkpoint(path: Path):
    models = build_models(torch.device("cpu"))
    hp = HyperParams()
    payload = {
        "models": {
            "policy_value": models["policy_value"].state_dict(),
            "sdf_fc1": models["sdf_fc1"].state_dict(),
        },
        "hyperparams": hp.__dict__,
    }
    torch.save(payload, path)
    return models, hp


def test_combined_checkpoint_loads_policy_and_sdf(tmp_path):
    ckpt = tmp_path / "combined.pt"
    _write_combined_checkpoint(ckpt)

    loaded = load_analysis_checkpoint(ckpt, device="cpu")

    assert loaded.metadata["checkpoint_format"] == "combined"
    assert loaded.metadata["policy_state_hash"]
    assert loaded.metadata["sdf_state_hash"]
    assert "policy_value" in loaded.models
    assert "sdf_fc1" in loaded.models


def test_raw_state_dict_requires_sdf_checkpoint(tmp_path):
    models = build_models(torch.device("cpu"))
    policy_path = tmp_path / "policy.pt"
    hp_path = tmp_path / "hp.json"
    torch.save(models["policy_value"].state_dict(), policy_path)
    hp_path.write_text(json.dumps(HyperParams().__dict__, default=str), encoding="utf-8")

    with pytest.raises(ValueError, match="requires sdf_checkpoint"):
        load_analysis_checkpoint(policy_path, hyperparams_json=hp_path, device="cpu")


def test_raw_state_dict_requires_hyperparams_unless_default_allowed(tmp_path):
    models = build_models(torch.device("cpu"))
    policy_path = tmp_path / "policy.pt"
    sdf_path = tmp_path / "sdf.pt"
    torch.save(models["policy_value"].state_dict(), policy_path)
    torch.save(models["sdf_fc1"].state_dict(), sdf_path)

    with pytest.raises(ValueError, match="requires hyperparams_json"):
        load_analysis_checkpoint(policy_path, sdf_checkpoint=sdf_path, device="cpu")


def test_fixed_slice_requires_calculated_macro_state(tmp_path):
    ckpt = tmp_path / "combined.pt"
    _write_combined_checkpoint(ckpt)

    with pytest.raises(ValueError, match="missing required fields"):
        evaluate_checkpoint_convergence_surfaces(
            [ckpt],
            b_grid=[0.0],
            z_grid=[0.0],
            state_mode="fixed_slice",
            fixed_state={"eta": 1, "i": 0, "x": 0, "hatcf": -2, "lnkf": 4},
            n_child_shocks=2,
            device="cpu",
        )


def test_firm_state_index_order_is_fixed():
    idx = FirmStateIndex()
    assert [idx.B, idx.Z, idx.ETA, idx.I, idx.X, idx.HATCF, idx.LNKF] == list(range(7))


def _child_bundle_from_episode_batch(batch):
    children = batch["children"]
    parent = batch["parent"]
    child = torch.stack([c[:, :7] for c in children], dim=1)
    m = torch.stack([c[:, 7:8] for c in children], dim=1)
    weights = torch.full((parent.shape[0], len(children)), 1.0 / len(children))
    return ChildExogenousBundle(
        z_next=child[:, :, 1:2],
        eta_next=child[:, :, 2:3],
        i_next=child[:, :, 3:4],
        x_next=child[:, :, 4:5],
        hatcf_next=child[:, :, 5:6],
        lnkf_next=child[:, :, 6:7],
        m_raw=m,
        branch_weights=weights,
    )


def test_vectorized_backend_matches_episode_reference_for_all_equations():
    episode = _episode()
    batch = _batch(episode.device)
    backend = VectorizedBellmanSurfaceBackend(
        episode.models["policy_value"],
        episode.hyperparams,
        p0_loss=episode.loss_fns["p0"],
        pi_loss=episode.loss_fns["pi"],
        q_loss=episode.loss_fns["q"],
    )
    child = _child_bundle_from_episode_batch(batch)

    with torch.no_grad():
        actual = backend.compute_signed_residuals(batch["parent"][:, :7], child)
        refs = {
            "p0": {
                "train": episode._compute_p0_bellman_signed_residuals(batch, m_mode="train"),
                "raw": episode._compute_p0_bellman_signed_residuals(batch, m_mode="raw"),
            },
            "pi": {
                "train": episode._compute_pi_bellman_signed_residuals(batch, m_mode="train"),
                "raw": episode._compute_pi_bellman_signed_residuals(batch, m_mode="raw"),
            },
            "q": {
                "train": episode._compute_q_bellman_signed_residuals(batch, m_mode="train"),
                "raw": episode._compute_q_bellman_signed_residuals(batch, m_mode="raw"),
            },
        }

    for eq in ("p0", "pi", "q"):
        for mode in ("train", "raw"):
            assert torch.allclose(actual[eq][mode], refs[eq][mode], atol=1e-6, rtol=1e-5)


def test_reduce_signed_surface_cancellation_and_realized_abs():
    signed = torch.tensor([[0.3, -0.3]])
    weights = torch.tensor([[0.5, 0.5]])

    reduced = reduce_signed_surface(signed, weights)

    assert reduced["conditional_abs"].item() == pytest.approx(0.0, abs=1e-7)
    assert reduced["realized_abs"].item() == pytest.approx(0.3, abs=1e-7)


def test_convergence_shock_bank_seed_is_deterministic_and_rng_neutral():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    a = ConvergenceShockBank.create(3, 2, seed=123, device="cpu")
    b = ConvergenceShockBank.create(3, 2, seed=123, device="cpu")

    assert torch.equal(a.eps_x, b.eps_x)
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_child_transition_depends_on_parent_x_z_and_uses_calculated_macro():
    class FakeSDF(torch.nn.Module):
        def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
            return (
                torch.ones_like(x_prev),
                torch.ones_like(x_curr),
                torch.ones_like(x_curr),
                hatcf_prev.unsqueeze(1).expand_as(x_curr) + 1.0,
                lnkf_prev.unsqueeze(1).expand_as(x_curr) + 1.0,
            )

    parent = torch.tensor([
        [0.1, 0.0, 1.0, 0.2, -2.0, -9.0, 9.0],
        [0.1, 1.0, 1.0, 0.2, -1.0, -8.0, 8.0],
    ])
    shock = ConvergenceShockBank.create(2, 2, seed=5, device="cpu")
    macro = MacroTransitionContext(
        hatc_cal=torch.full((2, 1), -2.1),
        lnk_cal=torch.full((2, 1), 4.1),
    )

    child = build_child_exogenous_bundle(FakeSDF(), parent, macro, shock)

    assert not torch.allclose(child.x_next[0], child.x_next[1])
    assert not torch.allclose(child.z_next[0], child.z_next[1])
    assert torch.allclose(child.hatcf_next, torch.full_like(child.hatcf_next, -1.1))
    assert torch.allclose(child.lnkf_next, torch.full_like(child.lnkf_next, 5.1))


def test_reference_distribution_requires_calculated_macro(tmp_path):
    ckpt = tmp_path / "combined.pt"
    _write_combined_checkpoint(ckpt)
    ref = pd.DataFrame({
        "b": [0.1], "z": [0.0], "eta": [1.0], "i": [0.1],
        "x": [0.0], "hatcf": [-2.0], "lnkf": [4.0],
    })

    with pytest.raises(ValueError, match="hatc_cal"):
        evaluate_checkpoint_convergence_surfaces(
            [ckpt],
            b_grid=[0.0],
            z_grid=[0.0],
            state_mode="reference_distribution",
            reference_data=ref,
            n_reference_states=1,
            n_child_shocks=2,
            device="cpu",
        )


def test_surface_api_is_deterministic_and_outputs_files(tmp_path):
    ckpt = tmp_path / "combined.pt"
    _write_combined_checkpoint(ckpt)
    fixed = {
        "eta": 1.0,
        "i": 0.1,
        "x": 0.0,
        "hatcf": -2.0,
        "lnkf": 4.0,
        "hatc_cal": -2.1,
        "lnk_cal": 4.1,
    }
    out = tmp_path / "surface"

    r1 = evaluate_checkpoint_convergence_surfaces(
        [ckpt],
        b_grid=[0.0, 0.5],
        z_grid=[-0.1, 0.1],
        state_mode="fixed_slice",
        fixed_state=fixed,
        n_child_shocks=2,
        seed=2026,
        device="cpu",
        output_dir=out,
        parent_chunk_size=2,
        child_chunk_size=2,
    )
    r2 = evaluate_checkpoint_convergence_surfaces(
        [ckpt],
        b_grid=[0.0, 0.5],
        z_grid=[-0.1, 0.1],
        state_mode="fixed_slice",
        fixed_state=fixed,
        n_child_shocks=2,
        seed=2026,
        device="cpu",
        parent_chunk_size=2,
        child_chunk_size=1,
    )

    assert np.allclose(
        r1.long_table["value"].to_numpy(),
        r2.long_table["value"].to_numpy(),
        atol=1e-3,
        rtol=1e-6,
        equal_nan=True,
    )
    assert (out / "manifest.json").exists()
    assert (out / "surface_long.csv").exists()
    assert (out / "raw_surface.pt").exists()
    assert len(list(out.glob("*.png"))) >= 3
    assert set(["checkpoint", "equation", "m_mode", "metric", "aggregation", "b", "z", "value"]).issubset(r1.long_table.columns)


def test_reference_distribution_aggregation_rows(tmp_path):
    ckpt = tmp_path / "combined.pt"
    _write_combined_checkpoint(ckpt)
    ref = pd.DataFrame({
        "b": [0.1, 0.2],
        "z": [0.0, 0.1],
        "eta": [1.0, 0.0],
        "i": [0.1, 0.2],
        "x": [0.0, 0.1],
        "hatcf": [-2.0, -2.2],
        "lnkf": [4.0, 4.2],
        "hatc_cal": [-2.1, -2.3],
        "lnk_cal": [4.1, 4.3],
    })

    result = evaluate_checkpoint_convergence_surfaces(
        [ckpt],
        b_grid=[0.0],
        z_grid=[0.0],
        state_mode="reference_distribution",
        reference_data=ref,
        n_reference_states=2,
        n_child_shocks=2,
        seed=2026,
        device="cpu",
    )

    assert {"mean", "p50", "p90", "p99", "max"}.issubset(set(result.long_table["aggregation"]))
    assert result.support_metadata["support_available"] is True
