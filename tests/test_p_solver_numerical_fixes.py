"""最终数值正确性修复验收（Prompt L1-L3 / P1-P3 / W1-W3 / F1-F4）。

L1-L3: L-BFGS Nonfinite Closure 的保守拒绝/回退。
P1-P3: Initial Validation 非有限的硬阻断。
W1-W3: L-BFGS Parent-Count 加权的数值等价性。
F1-F4: t0 Floor 归因必须由理论效应解释。
"""

from pathlib import Path
import math
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from config.hyperparams import HyperParams  # noqa: E402
from data.simulate_ts_parallel import _initial_macro_floor_decomposition  # noqa: E402
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


def _make_batch(device: torch.device, n_rows: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    parent_state = torch.rand(n_rows, 7, generator=generator) * 0.4 - 0.2
    parent_state[:, 2] = torch.tensor(
        [1.0 if i % 2 == 0 else 0.0 for i in range(n_rows)],
        device=device,
        dtype=torch.float32,
    )
    parent_state[:, 5] = parent_state[:, 5] - 2.0
    parent_state[:, 6] = parent_state[:, 6] + 4.0
    children = []
    for j in range(2):
        child = parent_state.clone()
        child[:, 1:2] = child[:, 1:2] + 0.05 * (j + 1)
        child[:, 2:3] = torch.tensor(
            [[1.0] if (i + j) % 2 == 0 else [0.0] for i in range(n_rows)],
            device=device,
            dtype=torch.float32,
        )
        m = torch.full((n_rows, 1), 0.98 + 0.04 * j, device=device)
        children.append(torch.cat([child, m], dim=1))
    return {
        "parent": torch.cat(
            [parent_state, torch.ones(n_rows, 1, device=device)], dim=1
        ),
        "children": children,
    }


def _slice_batch(batch, start: int, stop: int):
    return {
        "parent": batch["parent"][start:stop],
        "children": [child[start:stop] for child in batch["children"]],
    }


def _episode() -> Episode:
    device = torch.device("cpu")
    Config.DEVICE = device
    torch.manual_seed(777)
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


# --- L1-L3: L-BFGS Nonfinite Closure ----------------------------------------

def test_L1_nan_gradient_closure_rejected_and_iteration_rolled_back(monkeypatch):
    """Closure 梯度 NaN、最终参数仍 finite 时也必须 rejected_numerical。"""
    episode = _episode()
    model = episode.models["policy_value"]
    batch = _make_batch(episode.device, 4)
    start_hash = episode._state_dict_hash(model)
    episode.hyperparams.p_root_adam_steps = 0  # 跳过 Adam，单独测 L-BFGS
    root_params = solver.p_root_params(episode, model)
    handle = root_params[0].register_hook(lambda grad: grad * float("nan"))
    try:
        summary = episode._run_frozen_policy_p_stage(
            [batch], [batch], model, inner_index=1
        )
    finally:
        handle.remove()

    assert summary["status"] == "rejected_numerical"
    rejected = summary["rejected_iterations"][0]
    assert rejected["failure_kind"] == "lbfgs_nonfinite_closure"
    record = summary["p_policy_iterations"][0]
    counters = record["failure_detail"]["lbfgs_counters"]
    assert counters["LBFGS_NONFINITE_CLOSURES"] >= 1
    assert counters["LBFGS_NONFINITE_CLOSURE_ABORT"] >= 1
    # 模型恢复到 Iteration Start（全模型哈希一致）。
    assert episode._state_dict_hash(model) == start_hash
    assert record["rollback"]["restored_iteration_start_state"] is True


def test_L2_infinite_closure_loss_never_enters_line_search(monkeypatch):
    """Closure Loss=Inf：不继续 Strong Wolfe，不污染 Best State。"""
    episode = _episode()
    model = episode.models["policy_value"]
    batch = _make_batch(episode.device, 4)
    start_hash = episode._state_dict_hash(model)
    # Loss 与参数图相连且为 Inf：backward 可执行但 loss 非有限。
    monkeypatch.setattr(
        solver,
        "root_loss",
        lambda model_, batch_: next(model_.parameters()).sum()
        * torch.tensor(float("inf")),
    )
    problem, _info = solver.build_problem(
        episode,
        model,
        solver.BPGridTeacher.from_hyperparams(
            model, episode.loss_fns["p0"], episode.loss_fns["pi"], episode.hyperparams
        ),
        [batch],
        q_model=model,
    )
    params = solver.p_root_params(episode, model)
    history, best_state, best, stop_reason, counters = solver.run_lbfgs_chunked(
        model=model,
        params=params,
        train=problem,
        validation=problem,
        max_iter=4,
        chunk_iterations=2,
        history_size=5,
    )
    assert stop_reason == "lbfgs_nonfinite_closure"
    assert counters["LBFGS_NONFINITE_CLOSURES"] >= 1
    assert counters["LBFGS_NONFINITE_CLOSURE_ABORT"] >= 1
    assert history[-1]["phase"] == "lbfgs_nonfinite_closure_abort"
    assert history[-1]["restored_best_state"] is True
    # Best State 未被非有限结果污染：恢复后与入口状态一致。
    assert episode._state_dict_hash(model) == start_hash


def test_L3_adam_improvement_not_reported_when_lbfgs_closure_nonfinite(monkeypatch):
    """Adam 已改善后 L-BFGS 出现非有限 Closure：不得报 accepted_improved。"""
    episode = _episode()
    model = episode.models["policy_value"]
    batch = _make_batch(episode.device, 4)
    start_hash = episode._state_dict_hash(model)
    q_hash = episode._params_content_hash(episode._policy_value_stage_params("q"))
    bp_hash = episode._params_content_hash(episode._policy_value_stage_params("bp"))
    sdf_hash = episode._state_dict_hash(episode.models["sdf_fc1"])

    real_adam = solver.run_adam
    handles = []

    def _adam_then_hook(**kwargs):
        result = real_adam(**kwargs)
        assert result[3] > 0, "Adam 应产生真实改善步"
        root_params = solver.p_root_params(episode, model)
        handles.append(
            root_params[0].register_hook(lambda grad: grad * float("nan"))
        )
        return result

    monkeypatch.setattr(solver, "run_adam", _adam_then_hook)
    try:
        summary = episode._run_frozen_policy_p_stage(
            [batch], [batch], model, inner_index=1
        )
    finally:
        for handle in handles:
            handle.remove()

    assert summary["status"] == "rejected_numerical"
    rejected = summary["rejected_iterations"][0]
    assert rejected["failure_kind"] == "lbfgs_nonfinite_closure"
    record = summary["p_policy_iterations"][0]
    assert int(record["failure_detail"]["adam_steps_before_abort"]) > 0
    # 保守回退：恢复完整 Iteration Start，不提交半更新状态。
    assert episode._state_dict_hash(model) == start_hash
    assert record["rollback"]["restored_iteration_start_state"] is True
    # Q/BP/FC1/SDF 哈希全部不变。
    assert episode._params_content_hash(episode._policy_value_stage_params("q")) == q_hash
    assert episode._params_content_hash(episode._policy_value_stage_params("bp")) == bp_hash
    assert episode._state_dict_hash(episode.models["sdf_fc1"]) == sdf_hash


# --- P1-P3: Initial Validation 非有限硬阻断 ---------------------------------

def _nonfinite_residual_injection(monkeypatch, key: str, value: float):
    real = solver.residual_tensors

    def _patched(model, batch):
        result = dict(real(model, batch))
        poisoned = result[key].clone()
        poisoned[0, 0] = value
        result[key] = poisoned
        return result

    monkeypatch.setattr(solver, "residual_tensors", _patched)


def test_P1_nan_initial_p0_residual_blocks_before_adam(monkeypatch):
    episode = _episode()
    model = episode.models["policy_value"]
    batch = _make_batch(episode.device, 4)
    start_hash = episode._state_dict_hash(model)
    _nonfinite_residual_injection(monkeypatch, "p0", float("nan"))
    adam_calls = {"n": 0}

    def _adam_bomb(**kwargs):
        adam_calls["n"] += 1
        raise AssertionError("optimizer must not run on nonfinite initial validation")

    monkeypatch.setattr(solver, "run_adam", _adam_bomb)
    summary = episode._run_frozen_policy_p_stage(
        [batch], [batch], model, inner_index=1
    )
    assert adam_calls["n"] == 0
    assert summary["status"] == "rejected_numerical"
    rejected = summary["rejected_iterations"][0]
    assert rejected["failure_kind"] == "initial_validation_nonfinite"
    assert episode._state_dict_hash(model) == start_hash
    assert summary["p_policy_iterations"][0]["rollback"][
        "initial_state_was_legal"
    ] is False


def test_P2_inf_initial_pi_residual_blocks_before_adam(monkeypatch):
    episode = _episode()
    model = episode.models["policy_value"]
    batch = _make_batch(episode.device, 4)
    start_hash = episode._state_dict_hash(model)
    _nonfinite_residual_injection(monkeypatch, "pi", float("inf"))
    adam_calls = {"n": 0}

    def _adam_bomb(**kwargs):
        adam_calls["n"] += 1
        raise AssertionError("optimizer must not run on nonfinite initial validation")

    monkeypatch.setattr(solver, "run_adam", _adam_bomb)
    summary = episode._run_frozen_policy_p_stage(
        [batch], [batch], model, inner_index=1
    )
    assert adam_calls["n"] == 0
    assert summary["status"] == "rejected_numerical"
    assert summary["rejected_iterations"][0]["failure_kind"] == (
        "initial_validation_nonfinite"
    )
    assert episode._state_dict_hash(model) == start_hash


def test_P3_illegal_initial_blocks_firm_stage_and_q(monkeypatch):
    """非法 Initial 时即使后续存在 finite candidate 也不得 accepted_reverted，
    且上层 _run_policy_value_staged 不进入 Q。"""
    episode = _episode()
    episode.hyperparams.firm_inner_cycles = 3
    model = episode.models["policy_value"]
    batch = _make_batch(episode.device, 4)
    q_calls = {"n": 0}

    _nonfinite_residual_injection(monkeypatch, "p0", float("nan"))
    monkeypatch.setattr(
        episode,
        "_run_q_bootstrap_stage",
        lambda *a, **k: {"status": "accepted", "optimizer_steps": 0},
    )

    def _q_stub(*args, **kwargs):
        q_calls["n"] += 1
        return {"status": "accepted", "optimizer_steps": 7}

    monkeypatch.setattr(episode, "_run_q_regime_training", _q_stub)
    monkeypatch.setattr(episode, "_update_firm_target_now", lambda *_args: None)

    result = episode._run_policy_value_staged([batch], [batch], n_epochs=1)

    assert result["metadata"]["policy_value_stage_status"] == "failed_pq"
    assert q_calls["n"] == 0, "Initial 非有限时不得进入 Q 训练"
    pq = result["metadata"]["policy_value_evaluation_stage"]
    assert pq["status"] == "rejected_numerical"
    assert pq["rejected_iterations"][0]["failure_kind"] == (
        "initial_validation_nonfinite"
    )


# --- W1-W3: L-BFGS Parent-Count 加权 ----------------------------------------

def _grad_vector(episode, model, ep3_batches):
    teacher = solver.BPGridTeacher.from_hyperparams(
        model,
        episode.loss_fns["p0"],
        episode.loss_fns["pi"],
        episode.hyperparams,
    )
    problem, _info = solver.build_problem(
        episode, model, teacher, ep3_batches, q_model=model
    )
    params = solver.p_root_params(episode, model)
    loss = solver.full_problem_loss(model, problem)
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)
    flat = []
    for grad in grads:
        flat.append(
            grad.reshape(-1) if grad is not None else torch.zeros(1)
        )
    return float(loss.detach()), torch.cat(flat)


def test_W1_batch_partition_invariance_of_loss_and_gradient():
    episode = _episode()
    model = episode.models["policy_value"]
    full = _make_batch(episode.device, 10, seed=11)
    partitions = [
        [full],
        [_slice_batch(full, 0, 6), _slice_batch(full, 6, 10)],
        [
            _slice_batch(full, 0, 4),
            _slice_batch(full, 4, 8),
            _slice_batch(full, 8, 10),
        ],
    ]
    reference_loss, reference_grad = _grad_vector(episode, model, partitions[0])
    for split in partitions[1:]:
        loss, grad = _grad_vector(episode, model, split)
        assert loss == pytest_approx(reference_loss)
        assert torch.allclose(grad, reference_grad, atol=1e-6, rtol=1e-5)


def test_W2_unequal_batches_match_parent_concatenated_reference():
    episode = _episode()
    model = episode.models["policy_value"]
    full = _make_batch(episode.device, 10, seed=23)
    ref_batches = [_slice_batch(full, 0, 10)]
    unequal = [_slice_batch(full, 0, 7), _slice_batch(full, 7, 10)]
    ref_loss, ref_grad = _grad_vector(episode, model, ref_batches)
    loss, grad = _grad_vector(episode, model, unequal)
    assert loss == pytest_approx(ref_loss)
    assert torch.allclose(grad, ref_grad, atol=1e-6, rtol=1e-5)


def test_W3_small_tail_batch_does_not_change_objective():
    """4+4+2（尾部小 Batch）与整体 Batch 的目标一致。"""
    episode = _episode()
    model = episode.models["policy_value"]
    full = _make_batch(episode.device, 10, seed=37)
    whole_loss, whole_grad = _grad_vector(episode, model, [full])
    split_batches = [
        _slice_batch(full, 0, 4),
        _slice_batch(full, 4, 8),
        _slice_batch(full, 8, 10),
    ]
    loss, grad = _grad_vector(episode, model, split_batches)
    assert loss == pytest_approx(whole_loss)
    assert torch.allclose(grad, whole_grad, atol=1e-6, rtol=1e-5)


def pytest_approx(value, rel=1e-6, abs=1e-7):
    import pytest as _pytest

    return _pytest.approx(value, rel=rel, abs=abs)


# --- F1-F4: t0 Floor 归因必须由理论效应解释 ---------------------------------

def _decompose(
    *,
    c_raw,
    k_ledger,
    floor,
    floor_binding,
    hatc_bias=0.0,
    lnk_bias=0.0,
    strict_tol=1e-5,
    parity_tol=1e-5,
):
    import torch as _torch

    c = _torch.tensor([c_raw], dtype=_torch.float32)
    k = _torch.tensor([k_ledger], dtype=_torch.float32)
    hatcf0 = _torch.log(c / k) + hatc_bias  # raw 根（可注入偏差）
    lnkf0 = _torch.log(k) + lnk_bias
    # 生产 floored 变换给出的计算宏观。
    c_eff = c.clamp_min(floor) if floor_binding else c
    k_eff = k.clamp_min(floor) if floor_binding else k
    hatc_cal0 = _torch.log(c_eff / k_eff)
    lnk_cal0 = _torch.log(k_eff)
    return _initial_macro_floor_decomposition(
        hatcf0=hatcf0,
        lnkf0=lnkf0,
        hatc_cal0=hatc_cal0,
        lnk_cal0=lnk_cal0,
        c_raw=c,
        k_ledger=k,
        floor=floor,
        floor_flag=_torch.tensor([floor_binding], dtype=_torch.bool),
        raw_clearing_ok=_torch.tensor([True]),
        ledger_parity_ok=_torch.tensor([True]),
        strict_tolerance=strict_tol,
        parity_tolerance=parity_tol,
    )


def test_F1_no_floor_macro_consistent_is_match():
    result = _decompose(c_raw=2.0, k_ledger=3.0, floor=1e-4, floor_binding=False)
    assert bool(result["macro_match_ok"][0]) is True
    assert int(result["classification"][0]) == 0  # MATCH
    assert not bool(result["unexplained_mismatch"][0])


def test_F2_floor_fully_explains_gap():
    # C_raw 低于 floor：生产 hatc_cal 被 clamp，gap 完全来自 floor。
    result = _decompose(c_raw=1e-6, k_ledger=3.0, floor=1e-4, floor_binding=True)
    assert bool(result["macro_match_ok"][0]) is False
    assert abs(float(result["observed_hatc_gap"][0])) > 1e-5
    assert bool(result["floor_effect_explains"][0]) is True
    assert int(result["classification"][0]) == 1  # FLOOR_EXPLAINED_MISMATCH
    assert abs(float(result["unexplained_hatc_gap"][0])) <= 1e-5
    assert abs(float(result["unexplained_lnk_gap"][0])) <= 1e-5


def test_F3_floor_plus_extra_bias_is_unexplained():
    # Floor 生效但同时注入额外 Macro Bias：不能误标 FLOOR_EXPLAINED。
    result = _decompose(
        c_raw=1e-6, k_ledger=3.0, floor=1e-4, floor_binding=True, hatc_bias=0.5
    )
    assert bool(result["macro_match_ok"][0]) is False
    assert abs(float(result["unexplained_hatc_gap"][0])) > 1e-5
    assert bool(result["floor_effect_explains"][0]) is False
    assert int(result["classification"][0]) == 2  # UNEXPLAINED_MISMATCH
    assert bool(result["unexplained_mismatch"][0]) is True


def test_F4_floor_plus_wrong_lnk_belief_is_unexplained():
    # Floor 生效但 LnKF0 与真实资本账本不一致：非纯 Floor 偏移。
    result = _decompose(
        c_raw=1e-6, k_ledger=3.0, floor=1e-4, floor_binding=True, lnk_bias=1.0
    )
    assert abs(float(result["unexplained_lnk_gap"][0])) > 1e-5
    assert bool(result["floor_effect_explains"][0]) is False
    assert int(result["classification"][0]) == 2  # UNEXPLAINED_MISMATCH