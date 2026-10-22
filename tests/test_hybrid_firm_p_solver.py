"""Hybrid Firm Solver 验收测试（b291 P Policy Iteration + EP3 Q Staged）。

Test 1: P 动态梯度（Parent/Child P 参与梯度；策略快照/Q/FC1/SDF 零梯度）。
Test 2: Bellman Operator Parity（frozen-policy RHS 重建 EP3 生产 value_star）。
Test 3: P Policy Iteration（3 次真实迭代、Greedy 重求、真实 Optimizer Steps、
        Q 参数不变）。
Test 4: Q 分区域（Q0→QD→QS→Polish 依次执行，使用最新 P Snapshot）。
Test 5: 完整 Inner3（P iterations=9、Q blocks=3、BP head steps=0）。
Test 6: Frozen Macro（Inner3 前后 FC1/SDF 哈希一致）。
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
from training.bp_grid_teacher import BPGridTeacher  # noqa: E402
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
    hp.bp_distill_patience = 1
    hp.pv_eval_epochs = 1
    hp.policy_lr = 1e-3
    hp.q_require_zero_phase = False
    hp.q_require_default_phase = False
    hp.q_require_survival_phase = False
    # Hybrid P solver: tiny budgets keep the CPU tests fast.
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
    # 正式 Hybrid 配置：BP Head 不训练，仅保留只读诊断。
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
    torch.manual_seed(123)
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
    episode.evaluate_bellman_convergence = lambda *args, **kwargs: {
        "passed": True,
        "equations": {
            key: {
                "conditional_train_m": {
                    "mean": 0.01,
                    "p90": 0.02,
                    "passed": True,
                    "n_parent": 2,
                }
            }
            for key in ("p0", "pi", "q")
        },
    }
    episode.models["sdf_fc1"] = nn.Linear(1, 1, bias=False).to(device)
    return episode


def _teacher(episode: Episode) -> BPGridTeacher:
    return BPGridTeacher.from_hyperparams(
        episode.models["policy_value"],
        episode.loss_fns["p0"],
        episode.loss_fns["pi"],
        episode.hyperparams,
    )


def _set_head_bias(model: PolicyValueModel, value: float) -> None:
    """Shift the P0/PI output bias so child equity is positive or negative.

    正偏置 → 全部 Child 存活（Survival Child）；强负偏置 → 全部 Child 违约
    （Default Child）。用于覆盖 Test 2 两种 Child 情形并避免 survival=0
    在 Test 1 中阻断梯度。
    """
    with torch.no_grad():
        for head in (model.v0_head, model.vi_head):
            linears = [m for m in head.network.modules() if isinstance(m, nn.Linear)]
            linears[-1].bias.fill_(float(value))


# --- Test 1: P 动态梯度 -------------------------------------------------------

def test_p_dynamic_gradient_parent_and_child():
    episode = _episode()
    model = episode.models["policy_value"]
    _set_head_bias(model, 2.0)  # Survival Child：child equity 为正
    batch = _batch(episode.device)
    frozen = solver.build_frozen_policy_batch(
        episode, model, _teacher(episode), batch, batch_id=0
    )

    p_params = episode._policy_value_stage_params("p")
    q_params = episode._policy_value_stage_params("q")

    # Parent P 对 P 参数梯度非零。
    p0_online, pi_online = model._value_outputs(frozen.parent)
    grads_parent = torch.autograd.grad(
        (p0_online + pi_online).sum(), p_params, allow_unused=True
    )
    assert any(
        grad is not None and float(grad.abs().sum()) > 0.0 for grad in grads_parent
    )

    # Child P 对同一组 P 参数梯度非零（child equity 单独求梯度）。
    equity = solver.frozen_child_equity(model, frozen.p0)
    assert equity.requires_grad
    grads_child = torch.autograd.grad(
        equity.sum(), p_params, allow_unused=True, retain_graph=False
    )
    assert any(
        grad is not None and float(grad.abs().sum()) > 0.0 for grad in grads_child
    )

    # 冻结策略快照不接收梯度。
    for tensor in (
        frozen.p0.selector_pi,
        frozen.p0.survival,
        frozen.p0.cashflow,
        frozen.p0.bp_star,
        frozen.p0.child_weights,
        frozen.p0.m_used,
    ):
        assert not bool(tensor.requires_grad)

    # root_loss 反传后：Q 参数零梯度、FC1/SDF 零梯度。
    model.zero_grad(set_to_none=True)
    loss = solver.root_loss(model, frozen)
    assert loss.requires_grad
    loss.backward()
    assert all(param.grad is None for param in q_params)
    # P Solver 学习范围（value_encoder/v0_head/vi_head）真实收到梯度。
    root_params = solver.p_root_params(episode, model)
    root_ids = {id(p) for p in root_params}
    for param in root_params:
        assert param.grad is not None
        assert float(param.grad.abs().sum()) > 0.0
    # barz/bari 派生头不在 residual 图内，保持零梯度/零更新（Prompt 3.5）。
    for param in model.parameters():
        if id(param) not in root_ids and id(param) not in {id(p) for p in q_params}:
            if param.grad is not None:
                assert float(param.grad.abs().sum()) == 0.0
    sdf_params = list(episode.models["sdf_fc1"].parameters())
    assert all(param.grad is None for param in sdf_params)


# --- Test 2: Bellman Operator Parity -----------------------------------------

def test_bellman_operator_parity_reproduces_production_rhs():
    for bias, label in ((2.0, "survival_child"), (-2.0, "default_child")):
        episode = _episode()
        model = episode.models["policy_value"]
        _set_head_bias(model, bias)
        batch = _batch(episode.device)
        frozen = solver.build_frozen_policy_batch(
            episode, model, _teacher(episode), batch, batch_id=0
        )
        assert frozen.parity_passed, (
            f"{label}: parity error={frozen.parity_max_abs_error}"
        )

        # 分别检查 P0 / PI，逐行（含 eta=1 与 eta=0 parent）。
        for branch in (frozen.p0, frozen.pi):
            rhs_snapshot = branch.cashflow + branch.continuation_factor * (
                branch.child_weights * branch.m_used * solver.frozen_child_equity(
                    model, branch, detach_child=True
                )
            ).sum(dim=1, keepdim=True)
            # teacher 分解（cashflow_mean + continuation_mean）与 value_star 一致。
            teacher_rhs = (
                branch.production_cashflow_mean + branch.production_continuation_mean
            )
            assert torch.allclose(
                rhs_snapshot, branch.production_value, atol=1e-4, rtol=1e-4
            ), label
            assert torch.allclose(
                teacher_rhs, branch.production_value, atol=1e-4, rtol=1e-4
            ), label

        # 冻结快照的 child equity 与 EP3 生产 P(child) 相同（投资选择 + 存活
        # 掩码各计一次，不重复计权）。
        for branch in (frozen.p0, frozen.pi):
            equity_snapshot = solver.frozen_child_equity(
                model, branch, detach_child=True
            )
            v0, vi = solver._i_grid_values(model, branch.child_states)
            phat = torch.maximum(v0, vi).mean(dim=0)
            production_p_child = phat.clamp_min(0.0)
            assert torch.allclose(
                equity_snapshot, production_p_child, atol=1e-5, rtol=1e-5
            ), label
            # Survival/Default 掩码与 phat 符号一致。
            expected_survival = (phat > 0.0).to(phat.dtype)
            assert torch.equal(branch.survival, expected_survival), label


# --- Test 3: P Policy Iteration ----------------------------------------------

def test_three_real_p_policy_iterations_keep_q_frozen(monkeypatch):
    episode = _episode()
    model = episode.models["policy_value"]
    batch = _batch(episode.device)
    q_params = episode._policy_value_stage_params("q")
    q_hash_before = episode._params_content_hash(q_params)

    summary = episode._run_frozen_policy_p_stage(
        [batch], [batch], episode.models["policy_value"], inner_index=1
    )

    assert summary["status"] == "accepted"
    assert summary["p_solver"] == "frozen_policy_dynamic_residual"
    records = summary["p_policy_iterations"]
    assert len(records) == 3
    assert summary["p_policy_iteration_count"] == 3
    for record in records:
        # 真实 Optimizer Steps 非零；策略快照在单次 Evaluation 内冻结。
        assert int(record["optimizer_steps"]) > 0
        assert record["parity"]["train"]["parity_passed"] is True
        assert record["parity"]["validation"]["parity_passed"] is True
        # Greedy 重求与函数漂移被记录。
        assert "fresh_greedy_residual" in record
        assert "greedy_bp_drift" in record
        assert "function_drift" in record
    # Q 参数保持不变。
    assert episode._params_content_hash(q_params) == q_hash_before


# --- Test 4: Q 分区域 --------------------------------------------------------

def test_q_regime_phases_run_after_p_with_latest_snapshot(monkeypatch):
    episode = _episode()
    batch = _batch(episode.device)
    phases = []
    frozen_p_hashes = []

    def _record_phase(*, phase, batches, frozen_p_model, q_target_model, epochs,
                      validation_bank):
        phases.append(phase)
        frozen_p_hashes.append(episode._state_dict_hash(frozen_p_model))
        return {
            "phase": phase,
            "status": "accepted",
            "optimizer_steps": 1,
            "epochs": int(epochs),
        }

    monkeypatch.setattr(episode, "_run_q_regime_phase", _record_phase)
    monkeypatch.setattr(
        episode,
        "_run_q_bootstrap_stage",
        lambda *a, **k: {"status": "accepted", "optimizer_steps": 0},
    )
    monkeypatch.setattr(episode, "_update_firm_target_now", lambda *_args: None)

    result = episode._run_policy_value_staged([batch], [batch], n_epochs=1)

    assert result["metadata"]["policy_value_stage_status"] == "accepted"
    # Q 使用最新 P Snapshot：QD/QS 区域分类所用冻结 P 与 P 阶段结束后的
    # 在线 P 完全一致（不是上一 Inner 的过期 P）。
    assert phases == ["zero", "default", "survival", "polish"]
    online_hash = episode._state_dict_hash(episode.models["policy_value"])
    assert all(h == online_hash for h in frozen_p_hashes)
    # 不调用任何 b291 Q Picard Solver：Q 阶段记录全部来自 EP3 分区域训练。
    q_summary = result["metadata"]["q_regime_training_stage"]
    assert q_summary["status"] == "accepted"


# --- Test 5: 完整 Inner3 -----------------------------------------------------

def test_full_inner3_schedule_counts(monkeypatch):
    episode = _episode()
    episode.hyperparams.firm_inner_cycles = 3
    episode.hyperparams.p_policy_iterations_per_inner = 3
    batch = _batch(episode.device)

    q_blocks = {"count": 0}

    def _fake_q(*args, **kwargs):
        q_blocks["count"] += 1
        return {
            "status": "accepted",
            "q_stage_required_gate_passed": True,
            "optimizer_steps": 7,
            "phases": ["zero", "default", "survival", "polish"],
        }

    monkeypatch.setattr(episode, "_run_q_bootstrap_stage",
                        lambda *a, **k: {"status": "accepted", "optimizer_steps": 0})
    monkeypatch.setattr(episode, "_run_q_regime_training", _fake_q)
    monkeypatch.setattr(episode, "_update_firm_target_now", lambda *_args: None)

    result = episode._run_policy_value_staged([batch], [batch], n_epochs=1)

    metadata = result["metadata"]
    assert metadata["policy_value_stage_status"] == "accepted"
    cycles = metadata["inner_cycle_diagnostics"]
    assert len(cycles) == 3

    # P_POLICY_ITERATION_COUNT=9、Q_STAGED_BLOCK_COUNT=3。
    total_p_iterations = sum(
        int(cycle["p_policy_iteration_count"] or 0) for cycle in cycles
    )
    assert total_p_iterations == 9
    assert q_blocks["count"] == 3
    for cycle in cycles:
        assert int(cycle["p_policy_iteration_count"]) == 3
        assert int(cycle["q_optimizer_steps"]) == 7

    # 每轮真实更新步骤被记录。
    for cycle in cycles:
        assert int(cycle["p_optimizer_steps"]) > 0
        for record in cycle["p_policy_iterations"]:
            assert int(record["adam_steps"]) >= 1

    # BP_HEAD_OPTIMIZER_STEPS=0。
    bp_stage = metadata["bp_distillation_stage"]
    assert bp_stage["status"] == "diagnostic_only_training_disabled"
    assert bp_stage["bp_head_optimizer_steps"] == 0


# --- Test 6: Frozen Macro ----------------------------------------------------

def test_inner3_keeps_fc1_sdf_frozen(monkeypatch):
    episode = _episode()
    episode.hyperparams.firm_inner_cycles = 3
    batch = _batch(episode.device)
    monkeypatch.setattr(episode, "_run_q_bootstrap_stage",
                        lambda *a, **k: {"status": "accepted", "optimizer_steps": 0})
    monkeypatch.setattr(
        episode, "_run_q_regime_training",
        lambda *a, **k: {"status": "accepted", "q_stage_required_gate_passed": True,
                         "optimizer_steps": 7},
    )
    monkeypatch.setattr(episode, "_update_firm_target_now", lambda *_args: None)
    sdf_hash_before = episode._state_dict_hash(episode.models["sdf_fc1"])

    result = episode._run_policy_value_staged([batch], [batch], n_epochs=1)

    cycles = result["metadata"]["inner_cycle_diagnostics"]
    for cycle in cycles:
        assert cycle["sdf_fc1_hash"] == sdf_hash_before
        assert cycle["sdf_fc1_hash_after"] == sdf_hash_before
    assert episode._state_dict_hash(episode.models["sdf_fc1"]) == sdf_hash_before