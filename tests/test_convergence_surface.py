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
from analysis.economic_config import AnalysisEconomicConfig
from analysis.convergence_surface import (
    VectorizedBellmanSurfaceBackend,
    evaluate_checkpoint_convergence_surfaces,
    plot_fixed_grid_collection,
    reduce_signed_surface,
    _extract_default_boundary_rows,
    _fixed_boundary_status,
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
        "config_snapshot": AnalysisEconomicConfig.from_current_config().to_dict(),
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


def test_common_shocks_reused_across_grid_and_storage_not_grid_sized():
    bank = ConvergenceShockBank.create(2, 3, seed=123, device="cpu")
    n_grid = 4
    ref_index = torch.arange(2).repeat_interleave(n_grid)
    expanded = bank.gather(ref_index)

    assert bank.storage_numel == 4 * 2 * 3
    for ref in range(2):
        rows = expanded.eps_x[ref * n_grid:(ref + 1) * n_grid]
        assert torch.allclose(rows, rows[:1].expand_as(rows))
    assert not torch.allclose(expanded.eps_x[0], expanded.eps_x[n_grid])


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

    child = build_child_exogenous_bundle(
        FakeSDF(),
        parent,
        macro,
        shock,
        economic_config=AnalysisEconomicConfig.from_current_config(),
    )

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
        child_chunk_size=2,
    )

    assert np.allclose(
        r1.long_table["value"].to_numpy(),
        r2.long_table["value"].to_numpy(),
        atol=1e-6,
        rtol=1e-5,
        equal_nan=True,
    )
    assert (out / "manifest.json").exists()
    assert (out / "surface_long.csv").exists()
    assert (out / "raw_surface.pt").exists()
    assert len(list(out.glob("*.png"))) >= 3
    assert any(p.name.startswith("combined_") for p in out.glob("*.png"))
    assert set(["checkpoint", "equation", "m_mode", "metric", "aggregation", "b", "z", "value"]).issubset(r1.long_table.columns)
    assert r1.shock_bank_metadata["common_random_numbers"] is True
    assert r1.shock_bank_metadata["shock_bank_storage_numel"] == 4 * 1 * 2


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


def test_full_surface_api_is_rng_neutral_and_exception_safe(tmp_path):
    models = build_models(torch.device("cpu"))
    bad = tmp_path / "bad.pt"
    torch.save({"models": {"policy_value": models["policy_value"].state_dict(), "sdf_fc1": models["sdf_fc1"].state_dict()}, "hyperparams": HyperParams().__dict__}, bad)

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    with pytest.raises(ValueError, match="config_snapshot"):
        evaluate_checkpoint_convergence_surfaces(
            [bad],
            b_grid=[0.0],
            z_grid=[0.0],
            state_mode="fixed_slice",
            fixed_state={"eta": 1, "i": 0, "x": 0, "hatcf": -2, "lnkf": 4, "hatc_cal": -2, "lnk_cal": 4},
            n_child_shocks=2,
            device="cpu",
        )

    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_config_snapshot_overrides_current_config(tmp_path):
    ckpt = tmp_path / "combined.pt"
    _write_combined_checkpoint(ckpt)
    payload = torch.load(ckpt, map_location="cpu")
    payload["config_snapshot"]["RHO_X"] = 0.123
    torch.save(payload, ckpt)

    loaded = load_analysis_checkpoint(ckpt, device="cpu")

    assert loaded.economic_config.RHO_X == pytest.approx(0.123)
    assert loaded.metadata["config_source"] == "checkpoint"


def test_branch_weights_shapes_are_chunk_stable(tmp_path):
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
    kwargs = dict(
        checkpoint_paths=[ckpt],
        b_grid=[0.0, 0.5],
        z_grid=[0.0],
        state_mode="reference_distribution",
        reference_data=ref,
        n_reference_states=2,
        n_child_shocks=2,
        seed=2026,
        device="cpu",
    )

    a = evaluate_checkpoint_convergence_surfaces(**kwargs, branch_weights=torch.tensor([0.75, 0.25]), parent_chunk_size=1)
    b = evaluate_checkpoint_convergence_surfaces(**kwargs, branch_weights=torch.tensor([[0.75, 0.25], [0.25, 0.75]]), parent_chunk_size=2)
    c = evaluate_checkpoint_convergence_surfaces(**kwargs, branch_weights=torch.tensor([[0.75, 0.25], [0.75, 0.25], [0.25, 0.75], [0.25, 0.75]]), parent_chunk_size=2)

    assert len(a.long_table) == len(b.long_table) == len(c.long_table)
    assert np.allclose(b.long_table["value"].to_numpy(), c.long_table["value"].to_numpy(), equal_nan=True, atol=1e-6, rtol=1e-5)


def test_multi_checkpoint_outputs_do_not_overwrite_and_labels_are_recorded(tmp_path):
    ckpt1 = tmp_path / "one.pt"
    ckpt2 = tmp_path / "two.pt"
    _write_combined_checkpoint(ckpt1)
    _write_combined_checkpoint(ckpt2)
    out = tmp_path / "multi"
    fixed = {"eta": 1, "i": 0, "x": 0, "hatcf": -2, "lnkf": 4, "hatc_cal": -2.1, "lnk_cal": 4.1}

    result = evaluate_checkpoint_convergence_surfaces(
        [ckpt1, ckpt2],
        checkpoint_labels=["base", "other"],
        b_grid=[0.0, 0.5],
        z_grid=[0.0],
        state_mode="fixed_slice",
        fixed_state=fixed,
        n_child_shocks=2,
        device="cpu",
        output_dir=out,
    )

    assert set(result.long_table["checkpoint"]) == {"base", "other"}
    assert (out / "base_p0_conditional_train_value.png").exists()
    assert (out / "other_p0_conditional_train_value.png").exists()
    assert len(list(out.glob("delta_other_minus_base_*.png"))) >= 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert {m["label"] for m in manifest["checkpoint_metadata"]} == {"base", "other"}


def test_support_uses_reference_loo_not_grid_p90(tmp_path):
    ckpt = tmp_path / "combined.pt"
    _write_combined_checkpoint(ckpt)
    ref = pd.DataFrame({
        "b": [0.0, 0.01, 0.02],
        "z": [0.0, 0.01, 0.02],
        "eta": [1.0, 1.0, 1.0],
        "i": [0.1, 0.1, 0.1],
        "x": [0.0, 0.0, 0.0],
        "hatcf": [-2.0, -2.0, -2.0],
        "lnkf": [4.0, 4.0, 4.0],
        "hatc_cal": [-2.1, -2.1, -2.1],
        "lnk_cal": [4.1, 4.1, 4.1],
    })

    result = evaluate_checkpoint_convergence_surfaces(
        [ckpt],
        b_grid=[0.0, 100.0],
        z_grid=[0.0, 100.0],
        state_mode="reference_distribution",
        reference_data=ref,
        n_reference_states=3,
        n_child_shocks=2,
        device="cpu",
    )

    assert result.support_metadata["definition"] == "standardized_reference_loo_nearest_neighbor"
    assert sum(result.support_metadata["in_support"]) < 4


def test_fixed_cli_missing_params_reports_missing_list(tmp_path):
    from scripts.plot_convergence_surface import main

    with pytest.raises(SystemExit, match="fixed_slice is missing required arguments"):
        sys.argv = [
            "plot",
            "--checkpoint", str(tmp_path / "missing.pt"),
            "--state-mode", "fixed_slice",
            "--output-dir", str(tmp_path),
        ]
        main()


def test_dataframe_aliases_and_firm_macro_bundle(tmp_path):
    ckpt = tmp_path / "combined.pt"
    _write_combined_checkpoint(ckpt)
    firm = pd.DataFrame({
        "path": [0],
        "t": [0],
        "B": [0.1],
        "Z": [0.0],
        "ETA": [1.0],
        "I": [0.1],
        "X": [0.0],
        "Hatcf": [-2.0],
        "LnKF": [4.0],
    })
    macro = pd.DataFrame({"path": [0], "t": [0], "hatc_cal": [-2.1], "lnk_cal": [4.1]})

    result = evaluate_checkpoint_convergence_surfaces(
        [ckpt],
        b_grid=[0.0],
        z_grid=[0.0],
        state_mode="reference_distribution",
        reference_data={"firm": firm, "macro": macro},
        n_reference_states=1,
        n_child_shocks=2,
        device="cpu",
    )

    assert not result.long_table.empty


def test_reference_path_auto_joins_sibling_macro_pickle(tmp_path):
    ckpt = tmp_path / "combined.pt"
    _write_combined_checkpoint(ckpt)
    firm = pd.DataFrame({
        "path": [0, 0],
        "ID": [10, 10],
        "t": [0, 1],
        "branch": [-1, 0],
        "B": [0.1, 0.2],
        "Z": [0.0, 0.1],
        "ETA": [1.0, 0.0],
        "I": [0.1, 0.2],
        "X": [0.0, 0.1],
        "Hatcf": [-2.0, -9.0],
        "LnKF": [4.0, 9.0],
    })
    # Latest episode pkl writes true macro state in the sibling macro file as
    # Hatc/LnK. Some runs do not have a branch=-1 macro row, so the adapter must
    # fall back from (path,t,branch) to (path,t).
    macro = pd.DataFrame({"path": [0], "t": [0], "branch": [0], "Hatc": [-2.1], "LnK": [4.1]})
    firm_path = tmp_path / "ep1_stage_modeb.pkl"
    macro_path = tmp_path / "ep1_stage_modeb_macro.pkl"
    firm.to_pickle(firm_path)
    macro.to_pickle(macro_path)

    result = evaluate_checkpoint_convergence_surfaces(
        [ckpt],
        b_grid=[0.0],
        z_grid=[0.0],
        state_mode="reference_distribution",
        reference_data=firm_path,
        n_reference_states=1,
        n_child_shocks=2,
        device="cpu",
    )

    assert not result.long_table.empty
    assert result.manifests["context_metadata"]["n_reference"] == 1


def test_reference_adapter_coalesces_duplicate_columns_after_aliasing(tmp_path):
    ckpt = tmp_path / "combined.pt"
    _write_combined_checkpoint(ckpt)
    firm = pd.DataFrame({
        "path": [0],
        "t": [0],
        "branch": [-1],
        "b": [0.1],
        "z": [0.0],
        "ETA": [1.0],
        "i": [0.1],
        "I": [0.2],
        "x": [0.0],
        "Hatcf": [-2.0],
        "LnKF": [4.0],
    })
    macro = pd.DataFrame({"path": [0], "t": [0], "Hatc": [-2.1], "LnK": [4.1]})
    firm_path = tmp_path / "ep1_stage_modeb.pkl"
    macro_path = tmp_path / "ep1_stage_modeb_macro.pkl"
    firm.to_pickle(firm_path)
    macro.to_pickle(macro_path)

    result = evaluate_checkpoint_convergence_surfaces(
        [ckpt],
        b_grid=[0.0],
        z_grid=[0.0],
        state_mode="reference_distribution",
        reference_data=firm_path,
        n_reference_states=1,
        n_child_shocks=2,
        device="cpu",
    )

    assert not result.long_table.empty


def test_workload_guard_blocks_large_run(tmp_path):
    ckpt = tmp_path / "combined.pt"
    _write_combined_checkpoint(ckpt)

    with pytest.raises(ValueError, match="workload guard blocked"):
        evaluate_checkpoint_convergence_surfaces(
            [ckpt],
            b_grid=[0.0, 0.1],
            z_grid=[0.0, 0.1],
            state_mode="fixed_slice",
            fixed_state={"eta": 1, "i": 0, "x": 0, "hatcf": -2, "lnkf": 4, "hatc_cal": -2, "lnk_cal": 4},
            n_child_shocks=2,
            max_child_state_evals=1,
            device="cpu",
        )


def test_fixed_grid_outputs_full_domain_without_reference_or_support(tmp_path):
    ckpt = tmp_path / "combined.pt"
    _write_combined_checkpoint(ckpt)
    fixed = {"eta": 1, "i": 0.1, "x": 0.0, "hatcf": -2.0, "lnkf": 4.0, "hatc_cal": -2.1, "lnk_cal": 4.1}
    out = tmp_path / "fixed_grid"

    result = evaluate_checkpoint_convergence_surfaces(
        [ckpt],
        checkpoint_labels=["fg"],
        b_grid=[0.0, 0.5, 1.0],
        z_grid=[-1.0, 1.0],
        state_mode="fixed_grid",
        fixed_state=fixed,
        n_child_shocks=2,
        seed=2026,
        device="cpu",
        output_dir=out,
        include_signed=True,
        residual_threshold=0.001,
    )

    assert result.support_metadata["support_available"] is False
    assert len(result.long_table) == 1 * 3 * 3 * 2
    assert set(result.long_table["aggregation"]) == {"none"}
    assert {"conditional_signed", "conditional_abs", "phat", "default_probability", "survival_probability"}.issubset(result.long_table.columns)
    assert "mean" not in set(result.long_table["aggregation"])
    assert "p90" not in set(result.long_table["aggregation"])
    assert (out / "fg_p0_conditional_abs.png").exists()
    assert (out / "fg_pi_conditional_abs.png").exists()
    assert (out / "fg_q_conditional_abs.png").exists()
    assert (out / "fg_p0_conditional_signed.png").exists()
    assert (out / "default_boundary.csv").exists()
    assert not (out / "support_mask.png").exists()
    assert result.shock_bank_metadata["shock_bank_base_shape"] == [1, 2, 1]
    assert result.shock_bank_metadata["shock_bank_storage_numel"] == 4 * 1 * 2


def test_fixed_grid_chunk_size_invariance(tmp_path):
    ckpt = tmp_path / "combined.pt"
    _write_combined_checkpoint(ckpt)
    fixed = {"eta": 1, "i": 0.1, "x": 0.0, "hatcf": -2.0, "lnkf": 4.0, "hatc_cal": -2.1, "lnk_cal": 4.1}
    kwargs = dict(
        checkpoint_paths=[ckpt],
        b_grid=[0.0, 0.5],
        z_grid=[-0.5, 0.5],
        state_mode="fixed_grid",
        fixed_state=fixed,
        n_child_shocks=2,
        seed=2026,
        device="cpu",
    )

    a = evaluate_checkpoint_convergence_surfaces(**kwargs, parent_chunk_size=1, child_chunk_size=1)
    b = evaluate_checkpoint_convergence_surfaces(**kwargs, parent_chunk_size=8, child_chunk_size=8)

    assert np.allclose(
        a.long_table["conditional_signed"].to_numpy(),
        b.long_table["conditional_signed"].to_numpy(),
        atol=1e-6,
        rtol=1e-5,
        equal_nan=True,
    )


def test_fixed_grid_real_parent_and_child_chunking_records_forward_sizes(tmp_path, monkeypatch):
    ckpt = tmp_path / "combined.pt"
    _write_combined_checkpoint(ckpt)
    fixed = {"eta": 1, "i": 0.1, "x": 0.0, "hatcf": -2.0, "lnkf": 4.0, "hatc_cal": -2.1, "lnk_cal": 4.1}
    from models.policy_value import PolicyValueModel

    calls = []
    original = PolicyValueModel.forward

    def spy_forward(self, firm_state):
        calls.append(int(firm_state.shape[0]))
        return original(self, firm_state)

    monkeypatch.setattr(PolicyValueModel, "forward", spy_forward)

    evaluate_checkpoint_convergence_surfaces(
        [ckpt],
        b_grid=[0.0, 0.5],
        z_grid=[-0.5, 0.5],
        state_mode="fixed_grid",
        fixed_state=fixed,
        n_child_shocks=2,
        parent_chunk_size=2,
        child_chunk_size=3,
        seed=2026,
        device="cpu",
    )

    assert calls
    assert max(calls) <= 3
    assert 2 in calls
    assert 3 in calls
    assert len(calls) > 10


def test_default_boundary_helpers_export_all_components():
    b = np.linspace(0.0, 1.0, 5)
    z = np.linspace(-1.0, 1.0, 5)
    _, zz = np.meshgrid(b, z)
    boundary_grid = zz * zz - 0.25

    rows = _extract_default_boundary_rows(checkpoint="ck", b_values=b, z_values=z, boundary_grid=boundary_grid)

    assert _fixed_boundary_status(np.ones((2, 2))) == "all_survival"
    assert _fixed_boundary_status(-np.ones((2, 2))) == "all_default"
    assert _fixed_boundary_status(boundary_grid) == "observed"
    assert rows
    assert len({row["component_id"] for row in rows}) >= 2


def test_default_boundary_uses_default_probability_half_surface():
    b = np.linspace(0.0, 1.0, 5)
    z = np.linspace(-1.0, 1.0, 5)
    _, zz = np.meshgrid(b, z)
    default_probability = 0.5 + zz
    boundary_grid = 0.5 - default_probability

    rows = _extract_default_boundary_rows(checkpoint="ck", b_values=b, z_values=z, boundary_grid=boundary_grid)

    assert _fixed_boundary_status(boundary_grid) == "observed"
    assert rows
    assert max(abs(row["z"]) for row in rows) < 1e-6


def test_boundary_status_reports_partial_nonfinite():
    phat = np.array([[1.0, np.nan], [-1.0, 0.5]])
    assert _fixed_boundary_status(phat) == "partial_nonfinite"


def test_fixed_grid_workload_guard_reports_dimensions(tmp_path):
    ckpt = tmp_path / "combined.pt"
    _write_combined_checkpoint(ckpt)

    with pytest.raises(ValueError, match="n_b=3.*n_z=3.*n_child_shocks=4.*n_equations=3"):
        evaluate_checkpoint_convergence_surfaces(
            [ckpt],
            b_grid=[0.0, 0.5, 1.0],
            z_grid=[-1.0, 0.0, 1.0],
            state_mode="fixed_grid",
            fixed_state={"eta": 1, "i": 0.1, "x": 0.0, "hatcf": -2.0, "lnkf": 4.0, "hatc_cal": -2.1, "lnk_cal": 4.1},
            n_child_shocks=4,
            max_child_state_evals=1,
            device="cpu",
        )


def test_fixed_grid_slurm_uses_fixed_grid_not_reference_distribution():
    script = (ROOT / "slurm" / "run_convergence_surfaces_all_episodes_gpu.slurm").read_text()
    assert "--state-mode fixed_grid" in script
    assert "--reference-data" not in script
    assert "--n-reference-states" not in script
    assert 'Z_MIN="${Z_MIN:--2.0}"' in script
    assert 'Z_MAX="${Z_MAX:-2.0}"' in script
    assert "support_mask.png" not in script
    assert "FIXED_ETA" in script
    assert '"$PNG_COUNT" -lt 3' in script


def test_fixed_grid_collection_common_scale_and_mismatch_rejection(tmp_path):
    ckpt = tmp_path / "combined.pt"
    _write_combined_checkpoint(ckpt)
    fixed = {"eta": 1, "i": 0.1, "x": 0.0, "hatcf": -2.0, "lnkf": 4.0, "hatc_cal": -2.1, "lnk_cal": 4.1}
    dirs = []
    for label in ("ep0", "ep1"):
        out = tmp_path / label
        evaluate_checkpoint_convergence_surfaces(
            [ckpt],
            checkpoint_labels=[label],
            b_grid=[0.0, 0.5],
            z_grid=[-0.5, 0.5],
            state_mode="fixed_grid",
            fixed_state=fixed,
            n_child_shocks=2,
            seed=2026,
            device="cpu",
            output_dir=out,
        )
        dirs.append(out)

    common = tmp_path / "common"
    manifest = plot_fixed_grid_collection(dirs, common)

    assert "common_scale" in manifest
    assert (common / "ep0_p0_conditional_abs_common_scale.png").exists()
    assert (common / "ep1_q_conditional_abs_common_scale.png").exists()
    log_manifest = plot_fixed_grid_collection(dirs, tmp_path / "common_log", log_residual_scale=True)
    assert log_manifest["log_residual_scale"] is True
    assert (tmp_path / "common_log" / "ep0_p0_conditional_abs_common_scale.png").exists()

    bad_manifest = json.loads((dirs[1] / "manifest.json").read_text())
    bad_manifest["b_grid"] = [0.0, 0.1]
    (dirs[1] / "manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not comparable"):
        plot_fixed_grid_collection(dirs, tmp_path / "bad")
