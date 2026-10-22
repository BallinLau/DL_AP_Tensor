from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from analysis.hybrid_ep3_h10_unified import (
    HybridFresh100Contract,
    HybridMacroMixtureContract,
    assert_p_stage_isolation,
    assert_q_stage_isolation,
    assert_same_fresh100_pool,
    bp_disabled_stage_result,
    build_unified_fresh100_batches,
    configure_episode_for_hybrid,
    fresh100_tensor_to_macro_replay,
    hybrid_stage_plan,
    parameter_state_hash,
    pool_from_batches,
    run_ep3_q_regimes_on_unified_pool,
    run_h10_p_on_unified_pool,
    select_macro_mixture,
    stage_parameter_hashes,
    training_plan,
)
from analysis.run_hybrid_ep3_h10_unified import main as runner_main
from analysis.test_frozen_policy_root_solve import BP_PREFIXES, P_PREFIXES, Q_PREFIXES
from config import Config
from config.hyperparams import HyperParams
from data.finite_direct_longrun_replay import fit_edges
from data.tensor_data import TensorTable
from models.policy_value import PolicyValueModel
from training.episode import Episode


def _direct_model(seed: int = 7) -> PolicyValueModel:
    torch.manual_seed(seed)
    model = PolicyValueModel(
        share_hidden_dims=[4],
        share_output_dim=4,
        q_parameterization="direct",
    )
    model.eval()
    return model


def _episode():
    device = torch.device("cpu")
    hp = HyperParams()
    hp.policy_lr = 1e-3
    hp.q_zero_boundary_epochs = 1
    hp.q_default_pretrain_epochs = 1
    hp.q_survival_aio_epochs = 1
    hp.q_mixed_polish_epochs = 0
    online = _direct_model()
    target = _direct_model(8)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.AdamW(online.parameters(), lr=hp.policy_lr)
    return Episode(
        models={"policy_value": online},
        optimizers={"policy_value": optimizer},
        config=Config,
        hyperparams=hp,
        device=device,
        firm_target=target,
    )


def _configure_hybrid_episode():
    episode = _episode()
    contract = configure_episode_for_hybrid(episode)
    assert contract == dict(HybridFresh100Contract().__dict__)
    assert episode._simulation_bp_action_source() == "grid"
    assert episode._bp_head_training_enabled() is False
    return episode


def _tensor_table(rows: int = 12, branches: int = 2) -> TensorTable:
    columns = [
        "path", "t", "branch", "b", "z", "ETA", "i", "x", "Hatcf", "LnKF",
        "bp", "P", "Q", "M", "firm",
    ]
    torch.manual_seed(123)
    rows = max(3, rows)
    groups = rows // 3
    rows = groups * 3
    data = torch.zeros((rows, len(columns)), dtype=torch.float32)
    frame = pd.DataFrame(data.numpy(), columns=columns)
    for parent in range(rows // 3):
        for local, branch in enumerate((-1, 0, 1)):
            row = parent * 3 + local
            path = 1000 + parent
            time = 21 + parent
            if branch >= 0:
                time += 1
            frame.iloc[row, columns.index("path")] = path
            frame.iloc[row, columns.index("t")] = time
            frame.iloc[row, columns.index("branch")] = branch
            frame.iloc[row, columns.index("firm")] = 2000 + parent
    frame["b"] = np.linspace(0.05, 0.8, rows)
    frame["z"] = np.linspace(-0.4, 0.4, rows)
    frame["ETA"] = np.tile([0.0, 1.0], rows // 2)
    frame["i"] = np.linspace(0.0, 0.4, rows)
    frame["x"] = -2.0
    frame["Hatcf"] = np.linspace(-1.5, 1.0, rows)
    frame["LnKF"] = np.linspace(1.5, 3.5, rows)
    frame["bp"] = 0.3
    frame["P"] = 1.0
    frame["Q"] = 0.7
    frame["M"] = 0.98
    return TensorTable(torch.tensor(frame.to_numpy(dtype=np.float32)), columns)


def _macro_table(rows: int = 16) -> TensorTable:
    columns = ["path", "t", "branch", "Hatc", "LnK", "hatcf", "lnkf", "C", "K", "M", "n_firms"]
    frame = pd.DataFrame(index=range(rows), columns=columns, dtype=float)
    frame["path"] = np.repeat([1, 2, 3, 4], rows // 4)
    frame["t"] = np.tile([21, 22, 23, 24], rows // 4)
    frame["branch"] = 0
    frame["Hatc"] = np.linspace(-1.5, 1.0, rows)
    frame["LnK"] = np.linspace(1.5, 3.5, rows)
    frame["hatcf"] = frame["Hatc"] + 0.1
    frame["lnkf"] = frame["LnK"] + 0.1
    frame["C"] = np.exp(frame["hatcf"] + frame["lnkf"])
    frame["K"] = np.exp(frame["lnkf"])
    frame["M"] = 0.98
    frame["n_firms"] = 2
    return TensorTable(torch.tensor(frame.to_numpy(dtype=np.float32)), columns)


def test_hybrid_contract_and_stage_plan_are_exact():
    HybridFresh100Contract().validate()
    HybridMacroMixtureContract().validate()
    plan = training_plan()
    assert plan["outer_count"] == 2
    assert plan["fresh100"] == dict(HybridFresh100Contract().__dict__)
    assert plan["macro_mixture"]["on_policy_fraction"] == 0.80
    stages = hybrid_stage_plan()
    assert all(item["kind"] != "reuse_firm" for item in stages)
    assert all(item["kind"] != "firm_bank" for item in stages)
    assert sum(item["kind"] == "unified_firm_data" for item in stages) == 2
    assert sum(item["outer"] == 0 for item in stages) > 0
    assert sum(item["outer"] == 1 for item in stages) > 0


def test_hybrid_episode_uses_grid_and_disables_bp_head():
    episode = _configure_hybrid_episode()
    model = episode.models["policy_value"]
    result = bp_disabled_stage_result(episode, model)
    assert result["BP_HEAD_TRAINING_SKIPPED"] is True
    assert result["BP_HEAD_OPTIMIZER_STEPS"] == 0
    assert result["SIMULATION_BP_ACTION_SOURCE"] == "GRID"
    assert result["OPTIMIZER_EXECUTED"] is False
    assert result["BP_HASH_BEFORE"] == parameter_state_hash(model, BP_PREFIXES)


def test_unified_fresh100_pool_builds_same_train_batches_for_p_and_q():
    episode = _configure_hybrid_episode()
    table = _tensor_table()
    pool = build_unified_fresh100_batches(episode, table, batch_size=3)
    assert pool["train_batches"]
    assert pool["summary"]["enabled"] is True
    assert abs(pool["summary"]["requested_ratio"] - 0.20) < 1e-12
    rebuilt = pool_from_batches(pool["train_batches"])
    from analysis.hybrid_ep3_h10_unified import tensor_pool_fingerprint

    assert pool["fingerprint"] == tensor_pool_fingerprint(rebuilt)
    assert pool["pool"].parent.shape == rebuilt.parent.shape
    torch.testing.assert_close(pool["pool"].parent, rebuilt.parent)
    for left, right in zip(pool["pool"].children, rebuilt.children):
        torch.testing.assert_close(left, right)


def test_unified_fresh100_pool_mismatch_is_rejected():
    pool_a = SimpleNamespace(parent=torch.zeros((1, 8)), children=[torch.ones((1, 8))] * 2)
    pool_b = SimpleNamespace(parent=torch.ones((1, 8)), children=[torch.ones((1, 8))] * 2)
    from analysis.hybrid_ep3_h10_unified import tensor_pool_fingerprint

    with pytest.raises(ValueError, match="same unified Fresh100"):
        assert_same_fresh100_pool(tensor_pool_fingerprint(pool_a), tensor_pool_fingerprint(pool_b))


def test_parameter_isolation_guards():
    model = _direct_model()
    before = stage_parameter_hashes(model)
    after = dict(before)
    assert_p_stage_isolation(before, after)
    assert_q_stage_isolation(before, after)
    after["Q"] = "changed"
    with pytest.raises(RuntimeError, match="P stage changed Q"):
        assert_p_stage_isolation(before, after)
    after = dict(before)
    after["P"] = "changed"
    with pytest.raises(RuntimeError, match="Q stage changed P"):
        assert_q_stage_isolation(before, after)


def test_p_and_q_helpers_delegate_and_guard(monkeypatch):
    calls = {}

    def fake_p(**kwargs):
        calls["p"] = kwargs
        solved = kwargs["model"]
        with torch.no_grad():
            next(solved.v0_head.parameters()).add_(0.1)
        history = [{"step": 0}]
        return solved, {"p0": 1.0}, history

    def fake_q(self, batches, frozen_p, validation_batches=None):
        calls["q"] = dict(batches=batches, frozen_p=frozen_p, validation_batches=validation_batches)
        return {
            "status": "accepted",
            "optimizer_steps": 2,
            "phases": [{"phase": name} for name in ("zero", "default", "survival", "polish")],
        }

    monkeypatch.setattr("analysis.run_full_nested_equilibrium._p_policy_outer", fake_p)
    monkeypatch.setattr("training.episode.Episode._run_q_regime_training", fake_q)
    episode = _configure_hybrid_episode()
    model = episode.models["policy_value"]
    table = _tensor_table()
    pool = build_unified_fresh100_batches(episode, table, batch_size=3)
    solver_args = SimpleNamespace(
        p_adam_steps=1, p_adam_lr=1e-3, p_eval_every=1,
        p_lbfgs_max_iter=1, p_lbfgs_chunk_iterations=1,
    )
    solved, result = run_h10_p_on_unified_pool(
        model=model,
        episode=episode,
        pool=pool,
        batch_size=3,
        args=solver_args,
        policy_outer=0,
    )
    assert result["optimizer_executed"] is True
    assert calls["p"]["train_parents"].shape[0] > 0
    q_result = run_ep3_q_regimes_on_unified_pool(model=solved, episode=episode, pool=pool)
    assert q_result["optimizer_executed"] is True
    assert q_result["q_regime_phases"] == ["zero", "default", "survival", "polish"]
    assert q_result["unified_fresh100_fingerprint"] == pool["fingerprint"]
    assert calls["q"]["batches"] is pool["train_batches"]


def test_fresh100_macro_adapter_preserves_valid_rows_and_uses_burnin():
    frame = fresh100_tensor_to_macro_replay(_macro_table(), split="train")
    assert set(frame.split) == {"train"}
    assert (frame.parent_time >= 20).all()
    assert set(frame.child_branch) == {0}
    assert len(frame) > 0
    assert frame.economic_state_valid.all()

    with pytest.raises(ValueError, match="duplicated"):
        bad = _macro_table()
        bad.data[:, bad.columns.index("t")] = 21
        fresh100_tensor_to_macro_replay(bad, split="train")


def test_macro_mixture_is_80_20_and_validation_stays_separate():
    from test_h10_h100_replay import rows

    coverage = rows(100, t=1)
    replay = rows(30, start=1000, t=21)
    mixed, audit = select_macro_mixture(
        coverage=coverage,
        on_policy_replay=replay,
        edges=fit_edges([coverage, replay]),
        seed=99,
    )
    assert audit["LONGRUN"]["requested"] == min(30, int(0.8 * 4096))
    assert audit["COVERAGE"]["requested"] == 4096 - audit["LONGRUN"]["requested"]
    assert set(mixed.split) == {"train"}
    contract = HybridMacroMixtureContract(validation_mixed=True)
    with pytest.raises(ValueError, match="fixed H10 Validation"):
        contract.validate()


def test_runner_default_is_plan_only_and_creates_no_approval(tmp_path, capsys):
    sampling = tmp_path / "sampling"
    sampling.mkdir()
    for name in ("sampling_signature.json", "train_initial_bank.pt", "validation_initial_bank.pt"):
        (sampling / name).write_text("x" if not name.endswith(".json") else '{"fingerprint":"abc"}')
    checkpoint = tmp_path / "initial.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "output"
    code = runner_main([
        "--sampling-dir", str(sampling),
        "--initial-checkpoint", str(checkpoint),
        "--output", str(output),
    ])
    assert code == 0
    assert (output / "training_plan.json").exists()
    assert not list((output / "metadata").glob("approval*.json")) if (output / "metadata").exists() else True
    printed = capsys.readouterr().out
    assert "WAITING_FOR_USER_REVIEW" in printed
    assert '"formal_training_started": false' in printed.lower()
