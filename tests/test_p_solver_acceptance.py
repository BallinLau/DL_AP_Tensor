"""Hybrid P Solver 数值接受性验收（Prompt Tests B-E）。

Test B: Parity 失败不执行 Optimizer、不接受、参数完全不变。
Test C: NaN/Inf 残差、NaN 梯度、L-BFGS 非有限 closure、非有限 Fresh Greedy
        均不接受并回滚。
Test D: 零有效更新 → rejected_no_effective_update，原参数恢复。
Test E: 每个 Firm Inner 冻结当前 Online Q Snapshot（非旧 Firm Target Q），
        3 次迭代共享、Q 更新后下一 Inner 刷新。
"""

from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from config.hyperparams import HyperParams  # noqa: E402
from models.policy_value import PolicyValueModel  # noqa: E402
from training import frozen_policy_p_solver as solver  # noqa: E402
from training.episode import Episode  # noqa: E402


def _small_hyperparams() -> HyperParams:
    hp = HyperParams()
    hp.pv_bp_training_mode = "target_grid"
    hp.pv_training_flow = "staged"
    hp.firm_target_update = "stage_hard"
    hp.bp_grid_coarse_size = 3
    hp.bp_grid_refine_enabled = False
    hp.bp_grid_candidate_chunk_size = 2
    hp.bp_grid_parent_chunk_size = 2
    hp.bp_grid_max_expanded_states = 16
    hp.bp_grid_mix_policy_weight = 1.0
    hp.bp_grid_margin_scale = 1e-4
    hp.bp_grid_policy_loss_space = "logit"
    hp.bp_grid_logit_target_eps = 1e-4
    hp.pv_use_clipped_m = False
    hp.pv_target_grid_val_fraction = 0.0
    hp.bp_distill_epochs = 1
    hp.pv_eval_epochs = 1
    hp.policy_lr = 1e-3
    hp.q_require_zero_phase = False
    hp.q_require_default_phase = False
    hp.q_require_survival_phase = False
    hp.p_solver = "frozen_policy_dynamic_residual"
    hp.p_policy_iterations_per_inner = 3
    hp.p_root_adam_steps = 2
    hp.p_root_adam_lr = 1e-3
    hp.p_root_adam_clip = 10.0
    hp.p_root_eval_every = 1
    hp.p_root_lbfgs_max_iter = 2
    hp.p_root_lbfgs_chunk_iterations = 2
    hp.p_root_lbfgs_history_size = 5
    hp.q_solver = "ep3_regime_staged"
    hp.pv_bp_head_training_enabled = False
    return hp


def _batch(device: torch.device):
    parent_state = torch.tensor(
        [
            [0.10, 0.20, 1.00, 0.20, 0.10, -2.00, 4.00],
            [0.30, 0.10, 0.00, 0.40, 0.00, -1.80, 4.20],
        ],
        device=device,
        dtype=torch.float32,
    )
    child0 = parent_state.clone()
    child1 = parent_state.clone()
    child0[:, 1:2] = child0[:, 1:2] + 0.05
    child1[:, 1:2] = child1[:, 1:2] - 0.03
    child0[:, 2:3] = torch.tensor([[1.0], [0.0]], device=device)
    child1[:, 2:3] = torch.tensor([[0.0], [1.0]], device=device)
    m0 = torch.full((2, 1), 0.98, device=device)
    m1 = torch.full((2, 1), 1.02, device=device)
    return {
        "parent": torch.cat([parent_state, torch.ones(2, 1, device=device)], dim=1),
        "children": [
            torch.cat([child0, m0], dim=1),
            torch.cat([child1, m1], dim=1),
        ],
    }


def _episode() -> Episode:
    device = torch.device("cpu")
    Config.DEVICE = device
    torch.manual_seed(321)
    online = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
    target = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.AdamW(online.parameters(), lr=1e-3)
    episode = Episode(
        models={"policy_value": online},
        optimizers={"policy_value": optimizer},
        config=Config,
        hyperparams=_small_hyperparams(),
        device=device,
        firm_target=target,
    )
    episode._prepare_sdf_shock_bank_for_epoch = lambda *args, **kwargs: None
    episode.models["sdf_fc1"] = nn.Linear(1, 1, bias=False).to(device)
    return episode


def _q_hash(episode: Episode, model) -> str:
    params = episode._unique_params(
        [getattr(model, "q_encoder", None), getattr(model, "q_head", None)]
    )
    return episode._params_content_hash(params)


# --- Test B: P Parity Acceptance --------------------------------------------

def test_B_parity_failure_blocks_optimizer_and_rolls_back(monkeypatch):
    episode = _episode()
    model = episode.models["policy_value"]
    batch = _batch(episode.device)
    start_hash = episode._state_dict_hash(model)
    q_hash = episode._params_content_hash(episode._policy_value_stage_params("q"))
    sdf_hash = episode._state_dict_hash(episode.models["sdf_fc1"])

    real_build = solver.build_problem

    def _failing_build(episode_, model_, teacher_, batches, *, q_model=None):
        problem, info = real_build(episode_, model_, teacher_, batches, q_model=q_model)
        info = dict(info)
        info["parity_passed"] = False
        info["parity_failures"] = [
            {
                "batch_id": 0,
                "branch": "p0",
                "parity_max_abs_error": 0.5,
                "failing_parent_rows": [0],
                "failing_row_count": 1,
            }
        ]
        return problem, info

    adam_calls = {"n": 0}

    def _adam_bomb(**kwargs):
        adam_calls["n"] += 1
        raise AssertionError("P optimizer must not run on parity failure")

    monkeypatch.setattr(solver, "build_problem", _failing_build)
    monkeypatch.setattr(solver, "run_adam", _adam_bomb)

    summary = episode._run_frozen_policy_p_stage(
        [batch], [batch], model, inner_index=1
    )

    assert adam_calls["n"] == 0, "Parity 失败时不得调用 P Optimizer"
    assert summary["status"] == "rejected_numerical"
    assert summary["rejected_iterations"][0]["failure_kind"] == "parity_failed"
    failure = summary["p_policy_iterations"][0]["failure_detail"]
    assert failure["train_parity"]["parity_failures"][0]["branch"] == "p0"
    assert failure["train_parity"]["parity_failures"][0]["failing_parent_rows"] == [0]
    # P/Q/FC1/SDF 参数全部不变。
    assert episode._state_dict_hash(model) == start_hash
    assert episode._params_content_hash(episode._policy_value_stage_params("q")) == q_hash
    assert episode._state_dict_hash(episode.models["sdf_fc1"]) == sdf_hash


# --- Test C: P Nonfinite Acceptance -----------------------------------------

def _assert_rejected_and_restored(episode, model, summary, start_hash):
    assert summary["status"] != "accepted_improved", summary.get("failure_kind")
    assert summary["status"] in {
        "rejected_numerical",
        "rejected_no_effective_update",
        "accepted_reverted",
    }
    assert episode._state_dict_hash(model) == start_hash
    assert episode._params_content_hash(episode._policy_value_stage_params("q"))
    first = summary["p_policy_iterations"][0]
    if first.get("rollback"):
        assert first["rollback"].get("restored_iteration_start_state") or first[
            "rollback"
        ].get("kept_best_validation_state")


def test_C1_nan_loss_rejected_and_restored(monkeypatch):
    episode = _episode()
    model = episode.models["policy_value"]
    batch = _batch(episode.device)
    start_hash = episode._state_dict_hash(model)
    monkeypatch.setattr(
        solver, "root_loss", lambda *args, **kwargs: torch.tensor(float("nan"))
    )
    summary = episode._run_frozen_policy_p_stage(
        [batch], [batch], model, inner_index=1
    )
    _assert_rejected_and_restored(episode, model, summary, start_hash)
    assert summary["rejected_iterations"][0]["failure_kind"] == "adam_nonfinite_loss"


def test_C2_nan_gradient_rejected_without_step(monkeypatch):
    episode = _episode()
    model = episode.models["policy_value"]
    batch = _batch(episode.device)
    start_hash = episode._state_dict_hash(model)
    root_params = solver.p_root_params(episode, model)
    handle = root_params[0].register_hook(
        lambda grad: grad * float("nan")
    )
    try:
        summary = episode._run_frozen_policy_p_stage(
            [batch], [batch], model, inner_index=1
        )
    finally:
        handle.remove()
    _assert_rejected_and_restored(episode, model, summary, start_hash)
    assert summary["rejected_iterations"][0]["failure_kind"] == (
        "adam_nonfinite_gradient"
    )


def test_C3_lbfgs_nonfinite_closure_does_not_pollute_best(monkeypatch):
    episode = _episode()
    model = episode.models["policy_value"]
    batch = _batch(episode.device)
    start_hash = episode._state_dict_hash(model)
    episode.hyperparams.p_root_adam_steps = 0
    real_evaluate = solver.evaluate_problem_rows
    calls = {"n": 0}

    def _evaluate_after_first(model_, problem):
        calls["n"] += 1
        result, rows = real_evaluate(model_, problem)
        if calls["n"] >= 2:
            # 第二次起（L-BFGS chunk 验证）注入非有限结果。
            result = dict(result)
            result["all_finite"] = False
            result["nonfinite_count"] = 1
        return result, rows

    monkeypatch.setattr(solver, "evaluate_problem_rows", _evaluate_after_first)
    summary = episode._run_frozen_policy_p_stage(
        [batch], [batch], model, inner_index=1
    )
    _assert_rejected_and_restored(episode, model, summary, start_hash)


def test_C3b_lbfgs_chunk_restore_path(monkeypatch):
    episode = _episode()
    model = episode.models["policy_value"]
    batch = _batch(episode.device)
    start_hash = episode._state_dict_hash(model)
    real_evaluate = solver.evaluate_problem
    calls = {"n": 0}

    def _evaluate_after_first(model_, problem):
        calls["n"] += 1
        result = real_evaluate(model_, problem)
        if calls["n"] == 2:
            result = dict(result)
            result["all_finite"] = False
        return result

    monkeypatch.setattr(solver, "evaluate_problem", _evaluate_after_first)
    problem, _info = solver.build_problem(
        episode, model, solver.BPGridTeacher.from_hyperparams(
            model, episode.loss_fns["p0"], episode.loss_fns["pi"], episode.hyperparams
        ),
        [batch], q_model=model,
    )
    params = solver.p_root_params(episode, model)
    history, best_state, best, stop_reason, counters = solver.run_lbfgs_chunked(
        model=model,
        params=params,
        train=problem,
        validation=problem,
        max_iter=2,
        chunk_iterations=2,
        history_size=5,
    )
    assert stop_reason == "nonfinite_chunk_restored"
    assert history[-1]["restored_best_state"] is True
    assert counters["LBFGS_ITERATION_BUDGET"] == 2
    assert "LBFGS_CLOSURE_CALLS" in counters
    assert episode._state_dict_hash(model) == start_hash


def test_C4_nonfinite_fresh_greedy_rejected(monkeypatch):
    episode = _episode()
    model = episode.models["policy_value"]
    batch = _batch(episode.device)
    start_hash = episode._state_dict_hash(model)
    monkeypatch.setattr(
        solver,
        "_fresh_stats_from_rows",
        lambda *args, **kwargs: {
            "fresh_greedy_p0_norm_mae": float("nan"),
            "fresh_greedy_pi_norm_mae": float("nan"),
            "all_finite": False,
        },
    )
    summary = episode._run_frozen_policy_p_stage(
        [batch], [batch], model, inner_index=1
    )
    _assert_rejected_and_restored(episode, model, summary, start_hash)
    assert summary["rejected_iterations"][0]["failure_kind"] == (
        "fresh_greedy_nonfinite_initial"
    )


def test_C5_nonfinite_validation_rows_reject_best_state(monkeypatch):
    episode = _episode()
    model = episode.models["policy_value"]
    batch = _batch(episode.device)
    start_hash = episode._state_dict_hash(model)
    real_evaluate = solver.evaluate_problem_rows
    calls = {"n": 0}

    def _evaluate_nonfinite_after_initial(model_, problem):
        calls["n"] += 1
        result, rows = real_evaluate(model_, problem)
        if calls["n"] >= 2:
            # 首次（Initial Gate）保持合法；此后注入非有限行，覆盖
            # best-state 验收路径。
            result = dict(result)
            result["all_finite"] = False
            result["nonfinite_count"] = int(result.get("n_total", 1))
        return result, rows

    monkeypatch.setattr(solver, "evaluate_problem_rows", _evaluate_nonfinite_after_initial)
    summary = episode._run_frozen_policy_p_stage(
        [batch], [batch], model, inner_index=1
    )
    _assert_rejected_and_restored(episode, model, summary, start_hash)
    assert summary["rejected_iterations"][0]["failure_kind"] == (
        "best_state_nonfinite_validation"
    )


# --- Test D: No Effective Update --------------------------------------------

def test_D_zero_effective_update_not_accepted():
    episode = _episode()
    model = episode.models["policy_value"]
    batch = _batch(episode.device)
    start_hash = episode._state_dict_hash(model)
    episode.hyperparams.p_root_adam_steps = 0
    episode.hyperparams.p_root_lbfgs_max_iter = 0

    summary = episode._run_frozen_policy_p_stage(
        [batch], [batch], model, inner_index=1
    )

    assert summary["status"] == "rejected_no_effective_update"
    assert summary["rejected_iterations"][0]["failure_kind"] == (
        "no_effective_optimizer_step"
    )
    assert episode._state_dict_hash(model) == start_hash


# --- Test E: Current Q Snapshot ---------------------------------------------

def test_E_current_q_snapshot_is_online_q_not_stale_firm_target():
    episode = _episode()
    model = episode.models["policy_value"]
    batch = _batch(episode.device)

    # 人为让 firm_target.Q != online.Q。
    with torch.no_grad():
        for param in episode.firm_target.parameters():
            param.add_(0.7)
    q_online = _q_hash(episode, model)
    q_firm_target = _q_hash(episode, episode.firm_target)
    assert q_online != q_firm_target

    summary = episode._run_frozen_policy_p_stage(
        [batch], [batch], model, inner_index=1
    )

    # P Solver 使用 Online Q 的冻结 Snapshot，而非旧 Firm Target Q。
    assert summary["current_q_snapshot_hash"] == q_online
    assert summary["current_q_snapshot_hash"] != q_firm_target
    assert summary["current_q_online_hash"] == q_online
    # 三次 P Policy Iteration 共享同一冻结 Q，且 P 训练期间 Q 不变。
    records = summary["p_policy_iterations"]
    assert len(records) == 3
    for record in records:
        assert record["module_hashes"]["q_params"] == q_online

    # Q 更新后，下一 Inner 重新创建当前 Q Snapshot。
    with torch.no_grad():
        for param in model.q_encoder.parameters():
            param.add_(0.3)
        for param in model.q_head.parameters():
            param.add_(0.3)
    q_updated = _q_hash(episode, model)
    assert q_updated != q_online
    summary2 = episode._run_frozen_policy_p_stage(
        [batch], [batch], model, inner_index=2
    )
    assert summary2["current_q_snapshot_hash"] == q_updated
    assert summary2["current_q_snapshot_hash"] != q_online