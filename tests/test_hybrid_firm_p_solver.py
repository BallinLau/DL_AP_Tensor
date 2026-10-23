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

    assert summary["status"] in {
        "accepted",
        "accepted_with_reverts",
        "accepted_reverted",
    }, summary.get("failure")
    assert summary["p_solver"] == "frozen_policy_dynamic_residual"
    records = summary["p_policy_iterations"]
    assert len(records) == 3
    assert summary["p_policy_iteration_count"] == 3
    for record in records:
        # 真实 Optimizer Steps 非零；策略快照在单次 Evaluation 内冻结。
        assert int(record["effective_optimizer_steps"]) > 0
        assert record["status"] in {
            "accepted_improved",
            "accepted_reverted",
        }, record.get("failure_kind")
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
    epochs_seen = []
    frozen_p_hashes = []

    def _record_phase(*, phase, batches, frozen_p_model, q_target_model, epochs,
                      validation_bank):
        phases.append(phase)
        epochs_seen.append(int(epochs))
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
    # Test G: Q0/QD/QS/Polish 的 Epoch 不变（5/10/20/5）。
    assert epochs_seen == [5, 10, 20, 5]
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


# ===========================================================================
# P-only Parent η Balance (50/50) — Counterfactual Parent η Balance
# ===========================================================================


def _skewed_batch(device, n_rows=20, eta1_rows=1, seed=0):
    """Parent batch with an extremely skewed natural current-eta distribution."""
    g = torch.Generator().manual_seed(seed)
    parent_state = torch.zeros(n_rows, 7, device=device)
    parent_state[:, 0] = torch.rand(n_rows, generator=g) * 0.5
    parent_state[:, 1] = torch.randn(n_rows, generator=g) * 0.3
    parent_state[:, 2] = 0.0
    if eta1_rows > 0:
        parent_state[:eta1_rows, 2] = 1.0
    parent_state[:, 3] = torch.rand(n_rows, generator=g) * 0.4
    parent_state[:, 4] = torch.randn(n_rows, generator=g) * 0.1
    parent_state[:, 5] = -2.0 + torch.randn(n_rows, generator=g) * 0.2
    parent_state[:, 6] = 4.0 + torch.randn(n_rows, generator=g) * 0.1
    child0 = parent_state.clone()
    child1 = parent_state.clone()
    child0[:, 1:2] += 0.05
    child1[:, 1:2] -= 0.03
    child0[:, 2:3] = 1.0 - parent_state[:, 2:3]
    child1[:, 2:3] = parent_state[:, 2:3]
    m0 = torch.full((n_rows, 1), 0.98, device=device)
    m1 = torch.full((n_rows, 1), 1.02, device=device)
    return {
        "parent": torch.cat([parent_state, torch.ones(n_rows, 1, device=device)], dim=1),
        "children": [
            torch.cat([child0, m0], dim=1),
            torch.cat([child1, m1], dim=1),
        ],
        "source_id": torch.zeros(n_rows, dtype=torch.long, device=device),
    }


def _enable_eta_balance(episode, share=0.50):
    hp = episode.hyperparams
    hp.pv_current_eta_balance_enabled = True
    hp.pv_current_eta1_train_share = share
    hp.pv_current_eta_balance_validation = True
    hp.pv_current_eta_balance_seed = 97531
    hp.pv_eta_resample_enabled = False
    hp.pv_exact_eta_integration_enabled = True
    hp.bp_current_eta_resample_enabled = False
    # 测试中每次 P Policy Iteration 都执行 Natural 诊断（默认生产口径为
    # firm_inner_end，只在每个 Firm Inner 的最后一次 P 迭代执行一次）。
    hp.p_eta_natural_diag_frequency = "per_p_iteration"
    return hp


# --- Test 1: P Parent η = 50/50 ----------------------------------------------

def test_p_eta_balance_reaches_fifty_fifty_on_skewed_natural_sample():
    episode = _episode()
    _enable_eta_balance(episode)
    train = _skewed_batch(episode.device, n_rows=20, eta1_rows=1, seed=1)
    validation = _skewed_batch(episode.device, n_rows=10, eta1_rows=0, seed=2)
    train["source_id"][:10] = 0
    train["source_id"][10:] = 1
    original_parent = train["parent"].clone()
    original_children = [c.clone() for c in train["children"]]
    rng_before = torch.random.get_rng_state().clone()

    balanced_train, train_summary = episode._balance_p_current_eta_batches(
        [train], stream="train"
    )
    balanced_val, val_summary = episode._balance_p_current_eta_batches(
        [validation], stream="validation"
    )

    # RNG 不被意外消耗；重复调用完全确定。
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    repeat, _ = episode._balance_p_current_eta_batches([train], stream="train")
    assert torch.equal(balanced_train[0]["parent"], repeat[0]["parent"])

    # 比例约 50%（允许取整偏差），Parent/Child 数量对应。
    for summary in (train_summary, val_summary):
        assert abs(summary["eta1_share_after"] - 0.5) <= 0.01
        assert summary["eta0_count_after"] + summary["eta1_count_after"] == (
            summary["eta0_count"] + summary["eta1_count"]
        )
        assert summary["p_eta_sampling_method"] == "counterfactual_parent_eta_balance"
        assert summary["original_parent_child_preserved"] is True
    assert train_summary["eta1_share"] == 0.05  # 自然样本极偏
    assert val_summary["eta1_share"] == 0.0
    assert balanced_train[0]["parent"].shape[0] == 20
    assert balanced_val[0]["parent"].shape[0] == 10
    for child_b, child_o in zip(balanced_train[0]["children"], original_children):
        assert child_b.shape == child_o.shape

    # 不修改原始 Batch（仅 η 列在副本中重分配）。
    assert torch.equal(train["parent"], original_parent)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(train["children"], original_children)
    )

    # Train/Validation 没有泄漏：每个副本的行（除 η 列）与自己来源的 Batch 一致。
    assert torch.equal(
        balanced_train[0]["parent"][:, [0, 1, 3, 4, 5, 6, 7]],
        original_parent[:, [0, 1, 3, 4, 5, 6, 7]],
    )
    assert torch.equal(
        balanced_val[0]["parent"][:, [0, 1, 3, 4, 5, 6, 7]],
        validation["parent"][:, [0, 1, 3, 4, 5, 6, 7]],
    )
    # Source 级 η 比例与反事实重分配计数被如实记录；全局 50% 不得掩盖某个
    # Source 完全缺少 η=1 的情况（Source 级不做独立配平，只做透明报告）。
    sources = train_summary["source_eta_shares"]
    assert set(sources.keys()) == {"0", "1"}
    for stats in sources.values():
        assert stats["eta1_count_after"] >= 1
        assert 0.0 < stats["eta1_share_after"] < 1.0
    assert train_summary["reassigned_count"] > 0


# --- Test 2: Bellman Operator Parity（Dynamic P Solver，η=0/1）---------------

def test_dynamic_solver_eta_gate_parity_and_gradients_on_balanced_parents():
    episode = _episode()
    model = episode.models["policy_value"]
    _set_head_bias(model, 2.0)
    _enable_eta_balance(episode)
    batch = _skewed_batch(episode.device, n_rows=12, eta1_rows=0, seed=3)
    balanced, _ = episode._balance_p_current_eta_batches([batch], stream="train")
    balanced_parent = balanced[0]["parent"][:, :7]
    eta = balanced_parent[:, 2]
    eta0 = eta <= 0.5
    eta1 = eta > 0.5
    assert bool(eta0.any()) and bool(eta1.any())

    frozen = solver.build_frozen_policy_batch(
        episode, model, _teacher(episode), balanced[0], batch_id=0
    )
    # Frozen Policy Bellman RHS 与生产 Teacher 一致（parity 硬校验）。
    assert frozen.parity_passed, frozen.parity_max_abs_error

    b_parent = balanced_parent[:, 0:1]
    for branch in (frozen.p0, frozen.pi):
        child_b = branch.child_states[:, :, 0]
        # η=0：Grid BP 融资门控生效，下期债务保持 b_parent。
        assert torch.allclose(
            branch.bp_star[eta0], b_parent[eta0], atol=1e-6, rtol=0.0
        )
        assert torch.allclose(
            child_b[eta0],
            b_parent[eta0].expand_as(child_b[eta0]),
            atol=1e-6,
            rtol=0.0,
        )
        # η=1：下期债务使用最优 Grid BP。
        assert torch.allclose(
            child_b[eta1],
            branch.bp_star[eta1].expand_as(child_b[eta1]),
            atol=1e-6,
            rtol=0.0,
        )

    # Dynamic Residual 的 Parent 与 Child P 梯度均有效（同一 P scope）。
    p_params = solver.p_root_params(episode, model)
    model.zero_grad(set_to_none=True)
    loss = solver.root_loss(model, frozen)
    assert loss.requires_grad
    loss.backward()
    assert all(
        param.grad is not None and float(param.grad.abs().sum()) > 0.0
        for param in p_params
    )
    equity_child = solver.frozen_child_equity(
        model, frozen.p0, detach_child=False
    )
    grads_child = torch.autograd.grad(
        equity_child.sum(), p_params, allow_unused=True
    )
    assert any(
        grad is not None and float(grad.abs().sum()) > 0.0 for grad in grads_child
    )


# --- Test 3: Q 保持 Natural --------------------------------------------------

def test_inner3_p_uses_balanced_eta_and_q_keeps_natural(monkeypatch):
    episode = _episode()
    _enable_eta_balance(episode)
    episode.hyperparams.firm_inner_cycles = 3
    episode.hyperparams.p_policy_iterations_per_inner = 3
    train = _skewed_batch(episode.device, n_rows=20, eta1_rows=1, seed=4)
    validation = _skewed_batch(episode.device, n_rows=10, eta1_rows=0, seed=5)
    q_calls = []

    def _fake_q(batches, *args, **kwargs):
        q_calls.append(batches)
        return {
            "status": "accepted",
            "q_stage_required_gate_passed": True,
            "optimizer_steps": 7,
            "phases": ["zero", "default", "survival", "polish"],
        }

    monkeypatch.setattr(
        episode,
        "_run_q_bootstrap_stage",
        lambda *a, **k: {"status": "accepted", "optimizer_steps": 0},
    )
    monkeypatch.setattr(episode, "_run_q_regime_training", _fake_q)
    monkeypatch.setattr(episode, "_update_firm_target_now", lambda *_args: None)

    result = episode._run_policy_value_staged([train], [validation], n_epochs=1)
    metadata = result["metadata"]
    assert metadata["policy_value_stage_status"] == "accepted"

    # 9 次 P Policy Iteration 全部使用 Balanced Parent η（balanced 验证口径
    # 每层 η=0/η=1 各占一半）。
    cycles = metadata["inner_cycle_diagnostics"]
    assert len(cycles) == 3
    total_iterations = 0
    for cycle in cycles:
        for record in cycle["p_policy_iterations"]:
            total_iterations += 1
            frozen_p0 = record["frozen_policy_p0_residual"]
            assert frozen_p0["eta0_n"] == frozen_p0["eta1_n"] == 5.0
            assert record["natural_validation"] is not None
            assert record["natural_validation"]["eta1_share"] == 0.0
    assert total_iterations == 9

    # 3 次 Q Regime Training 使用原始 Natural η：对象未被替换、η 未被重分配。
    assert len(q_calls) == 3
    for batches in q_calls:
        assert batches[0] is train
        assert torch.equal(batches[0]["parent"], train["parent"])
    eta_summary = metadata["inner_cycle_diagnostics"][0]["p_eta_balance"]
    assert eta_summary["q_train_eta1_share_before"] == 0.05
    assert eta_summary["q_train_eta1_share_after"] == 0.05
    assert eta_summary["p_train_eta1_share_after"] == 0.5
    assert eta_summary["p_eta_sampling_method"] == "counterfactual_parent_eta_balance"

    # FC1/SDF 不更新；BP Head 不训练。
    for cycle in cycles:
        assert cycle["sdf_fc1_hash"] == cycle["sdf_fc1_hash_after"]
    bp_stage = metadata["bp_distillation_stage"]
    assert bp_stage["status"] == "diagnostic_only_training_disabled"
    assert bp_stage["bp_head_optimizer_steps"] == 0


# --- Test 4: Child η Exact Integration --------------------------------------

def test_child_eta_exact_integration_weights_untouched_by_balance():
    episode = _episode()
    _enable_eta_balance(episode)
    batch = _skewed_batch(episode.device, n_rows=6, eta1_rows=1, seed=6)
    children = batch["children"]
    _raw_m, m_list = episode._build_policy_m_lists(batch["parent"], children, 0.7, 1.3)
    _, _, _, weights_natural = episode._expand_policy_expectation_children(
        children, m_list, m_list
    )

    balanced, _ = episode._balance_p_current_eta_batches([batch], stream="train")
    b_children = balanced[0]["children"]
    _raw_m_b, m_list_b = episode._build_policy_m_lists(
        balanced[0]["parent"], b_children, 0.7, 1.3
    )
    expanded_b, _, _, weights_balanced = episode._expand_policy_expectation_children(
        b_children, m_list_b, m_list_b
    )

    # Parent η Balance 不改变 Child η_next 的期望权重。
    assert torch.equal(weights_natural, weights_balanced)
    assert weights_balanced is not None
    # 权重仍是 Config.ZETA 的 Bernoulli 概率（相邻 η 对：先 η=0 后 η=1），
    # 绝不会因 Parent 50/50 而变成 50/50。
    zeta = float(Config.ZETA)
    assert abs(zeta - 0.5) > 1e-6
    w = weights_balanced
    w_pairs = w.reshape(w.shape[0], -1, 2)
    base = w_pairs[:, :, 0] / (1.0 - zeta)
    assert torch.allclose(w_pairs[:, :, 1], base * zeta, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        w.sum(dim=-1), torch.ones(w.shape[0], device=w.device, dtype=w.dtype), atol=1e-6
    )
    # Child Shock Identity 与 M 口径未改变：扩展成员除 η_next 列外逐行与
    # 某个原始 child 完全一致（冲击/M 原样保留），η_next 恰好覆盖 {0,1}。
    keep = [c for c in range(children[0].shape[-1]) if c != 2]
    for entry in expanded_b:
        per_child = torch.stack(
            [
                (entry[:, keep] - original[:, keep]).abs().amax(dim=1)
                for original in children
            ],
            dim=0,
        )
        assert float(per_child.amin(dim=0).max()) == 0.0
    eta_next_all = torch.cat([entry[:, 2] for entry in expanded_b])
    assert set(eta_next_all.unique().tolist()) == {0.0, 1.0}


# --- Test 5: Validation 双口径 ------------------------------------------------

def test_dual_caliber_validation_reporting_and_revert_not_mislabeled(monkeypatch):
    episode = _episode()
    model = episode.models["policy_value"]
    _set_head_bias(model, 2.0)
    _enable_eta_balance(episode)
    train = _skewed_batch(episode.device, n_rows=12, eta1_rows=1, seed=7)
    validation = _skewed_batch(episode.device, n_rows=10, eta1_rows=0, seed=8)
    p_train, _ = episode._balance_p_current_eta_batches([train], stream="train")
    p_val, _ = episode._balance_p_current_eta_batches([validation], stream="validation")

    # 强制“无改善”路径：优化器保持初始状态但计为有效步，必须得到
    # accepted_reverted，且不得被误记为训练改善。
    def _stub_adam(**kwargs):
        eval0 = solver.evaluate_problem(kwargs["model"], kwargs["validation"])
        return [eval0], solver._clone_state(kwargs["model"]), dict(eval0), 1, "ok"

    def _stub_lbfgs(**kwargs):
        eval0 = solver.evaluate_problem(kwargs["model"], kwargs["validation"])
        counters = {
            "LBFGS_ITERATION_BUDGET": int(kwargs["max_iter"]),
            "LBFGS_ACTUAL_ITERATIONS": 1,
            "LBFGS_CLOSURE_CALLS": 0,
            "LBFGS_NONFINITE_CLOSURES": 0,
        }
        return [eval0], solver._clone_state(kwargs["model"]), dict(eval0), "max_iter_reached", counters

    monkeypatch.setattr(solver, "run_adam", _stub_adam)
    monkeypatch.setattr(solver, "run_lbfgs_chunked", _stub_lbfgs)

    summary = solver.run_p_policy_iteration(
        episode,
        model,
        train_batches=p_train,
        validation_batches=p_val,
        q_model=model,
        inner_index=1,
        iteration_index=1,
        adam_steps=1,
        adam_lr=1e-3,
        adam_clip=10.0,
        eval_every=1,
        lbfgs_max_iter=1,
        lbfgs_chunk_iterations=1,
        lbfgs_history_size=2,
        natural_validation_batches=[validation],
    )

    # Balanced Validation 口径：η=0/η=1 各 5 行，P0/PI 误差分层可独立读取。
    frozen_p0 = summary["frozen_policy_p0_residual"]
    frozen_pi = summary["frozen_policy_pi_residual"]
    for block in (frozen_p0, frozen_pi):
        assert block["eta0_n"] == 5.0
        assert block["eta1_n"] == 5.0
        assert block["eta0_mae"] is not None
        assert block["eta1_mae"] is not None
    # Fresh Greedy 同样分 η。
    fresh = summary["fresh_greedy_residual"]
    assert fresh["fresh_greedy_p0_norm_eta0_n_total"] == 5.0
    assert fresh["fresh_greedy_p0_norm_eta1_n_total"] == 5.0

    # Natural Validation 口径：未被覆盖（η1 share 保持自然 0.0），指标双出。
    natural = summary["natural_validation"]
    assert natural is not None
    assert natural["caliber"] == "natural_parent_eta"
    assert natural["eta1_share"] == 0.0
    assert natural["n_parents"] == 10
    assert natural["total_residual_initial"] is not None
    assert natural["total_residual_final"] is not None
    nat_final = natural["frozen_policy_final"]
    assert nat_final["p0_norm_eta0_n_total"] == 10.0
    assert nat_final["p0_norm_eta1_n_total"] == 0.0

    # 回退不得被误记为训练改善。
    assert summary["status"] == "accepted_reverted"
    assert summary["frozen_residual_improved"] is False
    assert summary["full_bellman_fixed_point_improvement"] is False
    assert natural["frozen_improved"] is False
    assert natural["natural_worsened"] is False