"""Natural Validation 诊断正确性回归测试（Codex Prompt §7 Test A-H）。

Test A：η Balance Off → 零重复诊断（Teacher 调用计数），复用主 Validation。
Test B：Validation 已为 Natural（balance on + validation off）→ 不重复计算。
Test C：η Balance On → Train/Validation 50/50、Original Mixture 不变、Q 原样。
Test D：诊断预算（Max Parents / Chunk Size / 固定子集 / 配对 / 可复现）。
Test E：诊断异常隔离（合法 P 更新不被误拒，Model/Optimizer/RNG 不变）。
Test F：三种 Validation 分布（Balanced / Original Mixture / Simulated-only）。
Test G：Natural Fresh Greedy 初始/终值与改善判断（三态）。
Test H：Old P / Hybrid P 求解器分流。
"""

from pathlib import Path
import sys

import pytest
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

NE = solver.NOT_EVALUABLE


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
                    "mean": 0.01, "p90": 0.02, "passed": True, "n_parent": 2,
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
    with torch.no_grad():
        for head in (model.v0_head, model.vi_head):
            linears = [m for m in head.network.modules() if isinstance(m, nn.Linear)]
            linears[-1].bias.fill_(float(value))


def _batch(device, n_rows=20, eta1_rows=1, seed=0, sources=None):
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
    if sources is None:
        sources = torch.zeros(n_rows, dtype=torch.long, device=device)
    return {
        "parent": torch.cat([parent_state, torch.ones(n_rows, 1, device=device)], dim=1),
        "children": [
            torch.cat([child0, m0], dim=1),
            torch.cat([child1, m1], dim=1),
        ],
        "source_id": sources.to(device),
    }


def _stub_noop_optimizers(monkeypatch):
    """优化器不改动模型但计 1 个有效步：得到 accepted_reverted 的合法路径。"""

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
        return (
            [eval0],
            solver._clone_state(kwargs["model"]),
            dict(eval0),
            "max_iter_reached",
            counters,
        )

    monkeypatch.setattr(solver, "run_adam", _stub_adam)
    monkeypatch.setattr(solver, "run_lbfgs_chunked", _stub_lbfgs)


def _run_iteration(episode, model, train, validation, **kwargs):
    params = dict(
        train_batches=train,
        validation_batches=validation,
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
    )
    params.update(kwargs)
    return solver.run_p_policy_iteration(episode, model, **params)


def _count_teacher_calls(monkeypatch):
    calls = {"compute_value_target": 0, "_evaluate_grid": 0, "build_problem": 0}
    orig_cvt = BPGridTeacher.compute_value_target
    orig_eg = BPGridTeacher._evaluate_grid
    orig_build = solver.build_problem

    def _cvt(self, *args, **kwargs):
        calls["compute_value_target"] += 1
        return orig_cvt(self, *args, **kwargs)

    def _eg(self, *args, **kwargs):
        calls["_evaluate_grid"] += 1
        return orig_eg(self, *args, **kwargs)

    def _build(*args, **kwargs):
        calls["build_problem"] += 1
        return orig_build(*args, **kwargs)

    monkeypatch.setattr(BPGridTeacher, "compute_value_target", _cvt)
    monkeypatch.setattr(BPGridTeacher, "_evaluate_grid", _eg)
    monkeypatch.setattr(solver, "build_problem", _build)
    return calls


def _fake_q_stage(monkeypatch, episode, q_calls=None):
    def _fake_q(batches, *args, **kwargs):
        if q_calls is not None:
            q_calls.append(batches)
        return {
            "status": "accepted",
            "q_stage_required_gate_passed": True,
            "optimizer_steps": 7,
            "phases": ["zero", "default", "survival", "polish"],
        }

    monkeypatch.setattr(
        episode, "_run_q_bootstrap_stage",
        lambda *a, **k: {"status": "accepted", "optimizer_steps": 0},
    )
    monkeypatch.setattr(episode, "_run_q_regime_training", _fake_q)
    monkeypatch.setattr(episode, "_update_firm_target_now", lambda *_args: None)


# --- Test A：η Balance Off → 零重复诊断 --------------------------------------

def test_a_eta_balance_off_reuses_validation_without_extra_teacher_calls(monkeypatch):
    episode = _episode()
    model = episode.models["policy_value"]
    _set_head_bias(model, 2.0)
    train = _batch(episode.device, n_rows=10, eta1_rows=1, seed=11)
    validation = _batch(episode.device, n_rows=8, eta1_rows=0, seed=12)
    _stub_noop_optimizers(monkeypatch)
    calls = _count_teacher_calls(monkeypatch)

    # Run 1：reuse_validation（Case A 语义）。
    summary_reuse = _run_iteration(
        episode, model, [train], [validation],
        natural_diag_mode="reuse_validation",
    )
    run1 = dict(calls)
    # Run 2：完全关闭诊断的基线。
    summary_none = _run_iteration(
        episode, model, [train], [validation],
        natural_diag_mode="not_evaluable",
    )
    run2 = dict(calls)
    delta = {key: run2[key] - run1[key] for key in calls}

    # 复用模式的 Grid Teacher / build_problem 调用次数与关闭诊断的基线完全
    # 相同：零重复 Bellman Grid Evaluation。
    assert run1 == delta, f"reuse adds zero extra calls: run1={run1} baseline={delta}"
    # 每次迭代只构建 train + validation 两份问题，绝无第三份 Natural 问题。
    assert run1["build_problem"] == 2

    natural = summary_reuse["natural_validation"]
    assert natural["natural_validation_reused"] is True
    assert natural["p_eta_sampling_method"] == "natural_parent_eta"
    assert natural["caliber"] == "natural_parent_eta"
    assert natural["diagnostic_status"] == "completed"
    assert natural["sampling"]["sampling_method"] == "reused_main_validation_diagnostics"
    # Acceptance 不受诊断影响。
    assert summary_reuse["status"] == summary_none["status"]
    assert summary_reuse["status"] in {"accepted_improved", "accepted_reverted"}

    # Episode 侧（PV_CURRENT_ETA_BALANCE_ENABLED=0）路由到 reuse。
    episode2 = _episode()
    episode2.hyperparams.pv_current_eta_balance_enabled = False
    episode2.hyperparams.firm_inner_cycles = 1
    episode2.hyperparams.p_policy_iterations_per_inner = 1
    _fake_q_stage(monkeypatch, episode2)
    train2 = _batch(episode2.device, n_rows=10, eta1_rows=1, seed=13)
    val2 = _batch(episode2.device, n_rows=8, eta1_rows=0, seed=14)
    result = episode2._run_policy_value_staged([train2], [val2], n_epochs=1)
    record = result["metadata"]["inner_cycle_diagnostics"][0]["p_policy_iterations"][0]
    nat = record["natural_validation"]
    assert result["metadata"]["policy_value_stage_status"] == "accepted"
    assert nat["natural_validation_reused"] is True
    assert nat["p_eta_sampling_method"] == "natural_parent_eta"
    assert nat["diagnostic_mode"] == "reuse_validation"


# --- Test B：Validation 已为 Natural（Case C）--------------------------------

def test_b_natural_validation_no_duplicate_problem_or_expansion(monkeypatch):
    episode = _episode()
    hp = episode.hyperparams
    hp.pv_current_eta_balance_enabled = True
    hp.pv_current_eta1_train_share = 0.50
    hp.pv_current_eta_balance_validation = False
    model = episode.models["policy_value"]
    _set_head_bias(model, 2.0)
    train = _batch(episode.device, n_rows=12, eta1_rows=2, seed=21)
    validation = _batch(episode.device, n_rows=10, eta1_rows=1, seed=22)
    original_parent = validation["parent"].clone()
    _stub_noop_optimizers(monkeypatch)

    expand_calls = {"n": 0}
    orig_expand = solver._expand_i_grid_states

    def _expand(*args, **kwargs):
        expand_calls["n"] += 1
        return orig_expand(*args, **kwargs)

    monkeypatch.setattr(solver, "_expand_i_grid_states", _expand)
    calls = _count_teacher_calls(monkeypatch)

    summary = _run_iteration(
        episode, model, [train], [validation],
        natural_diag_mode="reuse_validation",
        natural_calibers_identical=True,
    )
    run1 = dict(calls)
    expand_run1 = expand_calls["n"]
    baseline_summary = _run_iteration(
        episode, model, [train], [validation],
        natural_diag_mode="not_evaluable",
    )
    run2 = dict(calls)
    delta = {key: run2[key] - run1[key] for key in calls}

    # 不重复计算：Expanded Child States / Teacher 调用与无诊断基线一致。
    assert run1 == delta
    assert expand_calls["n"] - expand_run1 == expand_run1
    natural = summary["natural_validation"]
    baseline = baseline_summary["natural_validation"]
    assert natural["natural_validation_reused"] is True
    assert natural["calibers_identical"] is True
    assert baseline["diagnostic_status"] == NE
    # Parent η 的真实比例保持不变（natural 分布未被重采样）。
    expected_share = float((original_parent[:, 2] > 0.5).float().mean())
    assert natural["eta1_share"] == pytest.approx(expected_share)
    assert torch.equal(validation["parent"], original_parent)


# --- Test C：η Balance On（Case B）-------------------------------------------

def test_c_eta_balance_on_dual_caliber_and_natural_mixture(monkeypatch):
    episode = _episode()
    hp = episode.hyperparams
    hp.pv_current_eta_balance_enabled = True
    hp.pv_current_eta1_train_share = 0.50
    hp.pv_current_eta_balance_validation = True
    hp.pv_current_eta_balance_seed = 97531
    hp.firm_inner_cycles = 1
    hp.p_policy_iterations_per_inner = 1
    hp.p_eta_natural_diag_frequency = "per_p_iteration"
    model = episode.models["policy_value"]
    _set_head_bias(model, 2.0)
    train = _batch(episode.device, n_rows=20, eta1_rows=1, seed=31)
    validation = _batch(
        episode.device, n_rows=10, eta1_rows=0, seed=32,
        sources=torch.tensor([0] * 6 + [1] * 4),
    )
    original_parent = validation["parent"].clone()
    q_calls = []
    _fake_q_stage(monkeypatch, episode, q_calls=q_calls)

    result = episode._run_policy_value_staged([train], [validation], n_epochs=1)
    metadata = result["metadata"]
    assert metadata["policy_value_stage_status"] == "accepted"

    cycle = metadata["inner_cycle_diagnostics"][0]
    record = cycle["p_policy_iterations"][0]
    natural = record["natural_validation"]
    # P Train / Balanced Validation 为 50/50。
    frozen_p0 = record["frozen_policy_p0_residual"]
    assert frozen_p0["eta0_n"] == frozen_p0["eta1_n"]
    assert natural is not None
    # 独立 Natural 诊断：Original Mixture 保持自然 η 分布（这里 η1=0）。
    assert natural["diagnostic_status"] == "completed"
    assert natural["natural_validation_reused"] is False
    assert natural["diagnostic_mode"] == "independent"
    assert natural["eta1_share"] == 0.0
    views = natural["views"]
    assert views["original_mixture"]["eta1_share"] == 0.0
    assert views["balanced_validation"]["eta1_share"] == pytest.approx(0.5)
    # Simulated-only 视图（source0=6 行）。
    assert views["simulated_only"]["n_parents"] == 6
    assert views["simulated_only"]["source"] == "simulated_only_source0"
    # Q 使用原始 batch（η 未被重分配）。
    assert len(q_calls) == 1
    assert q_calls[0][0] is train
    assert torch.equal(q_calls[0][0]["parent"], train["parent"])
    # Original Validation 未被改写。
    assert torch.equal(validation["parent"], original_parent)
    # Child η Exact Integration 保持 Config.ZETA。
    assert abs(float(Config.ZETA) - 0.5) > 1e-6


# --- Test D：诊断预算 --------------------------------------------------------

def test_d_diagnostic_budget_caps_parents_chunks_and_preserves_pairing():
    episode = _episode()
    model = episode.models["policy_value"]
    _set_head_bias(model, 2.0)
    teacher = _teacher(episode)
    batches = [
        _batch(episode.device, n_rows=20, eta1_rows=2, seed=41 + i,
               sources=torch.tensor([0] * 10 + [1] * 10))
        for i in range(3)
    ]
    total_rows = sum(int(b["parent"].shape[0]) for b in batches)

    rng_before = torch.random.get_rng_state().clone()
    state = solver.natural_diag_prepare(
        episode, model, teacher, batches,
        q_model=model, max_parents=10, chunk_size=4, seed=97531,
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert state["status"] == "ready"
    sampling = state["sampling"]
    assert sampling["requested_parents"] == total_rows
    assert sampling["selected_parents"] == 10
    assert sampling["selected_parents"] <= sampling["max_parents"]
    assert 0.0 < sampling["sampling_ratio"] < 1.0
    assert sampling["stratified"] is False
    assert sampling["eta_rebalanced"] is False

    # Parent/Child 配对完整：subset 里每个 child 行都来自同一行号的 parent。
    subset_rows_total = 0
    for sub in state["subset"]:
        n_rows = int(sub["parent"].shape[0])
        subset_rows_total += n_rows
        assert all(int(c.shape[0]) == n_rows for c in sub["children"])
        # child 内容与其 parent 行的派生关系保持（z 列 ± 常数偏移）。
        assert torch.allclose(
            sub["children"][0][:, 1], sub["parent"][:, 1] + 0.05, atol=1e-6
        )
        assert torch.allclose(
            sub["children"][1][:, 1], sub["parent"][:, 1] - 0.03, atol=1e-6
        )
    assert subset_rows_total == 10

    # Chunk Size 受限：每个冻结 batch 的 Parent 行数 ≤ chunk_size。
    assert all(int(f.parent.shape[0]) <= 4 for f in state["problem_cpu"])

    # 可复现：同 seed 重跑选择同一子集。
    state2 = solver.natural_diag_prepare(
        episode, model, teacher, batches,
        q_model=model, max_parents=10, chunk_size=4, seed=97531,
    )
    for a, b in zip(state["subset"], state2["subset"]):
        assert torch.equal(a["parent"], b["parent"])

    # 同一固定子集用于更新前后比较（初始/终值行数与 η/source 一致）。
    block = solver.natural_diag_finalize(
        state, episode, model, teacher, q_model=model
    )
    assert block["diagnostic_n_parents"] == 10
    assert block["diagnostic_status"] == "completed"
    assert int(state["frozen_init"]["eta"].numel()) == 10
    assert block["frozen_policy_final"]["n_total"] == 10
    assert torch.equal(
        state["frozen_init"]["source"],
        torch.cat([s["source_id"].reshape(-1) for s in state["subset"]]),
    )
    # 前后比较基于同一子集：诊断块的 source 计数与抽样元数据一致。
    assert block["diagnostic_source_counts"] == {
        "source0": sampling["source_counts"]["source0"],
        "source1": sampling["source_counts"]["source1"],
        "unknown": sampling["source_counts"]["unknown"],
    }


# --- Test E：诊断异常隔离 ----------------------------------------------------

def test_e_diagnostic_exception_isolated_from_acceptance(monkeypatch):
    episode = _episode()
    model = episode.models["policy_value"]
    _set_head_bias(model, 2.0)
    train = _batch(episode.device, n_rows=12, eta1_rows=2, seed=51)
    validation = _batch(episode.device, n_rows=10, eta1_rows=1, seed=52)
    _stub_noop_optimizers(monkeypatch)

    def _boom(*args, **kwargs):
        raise ValueError("injected diagnostic failure")

    monkeypatch.setattr(solver, "natural_diag_finalize", _boom)

    rng_before = torch.random.get_rng_state().clone()
    optimizer_state_before = {
        k: v for k, v in episode.optimizers["policy_value"].state_dict().items()
        if k != "state"
    }
    summary = _run_iteration(
        episode, model, [train], [validation],
        natural_validation_batches=[validation],
        natural_diag_mode="independent",
        natural_diag_config={"max_parents": 8, "chunk_size": 4, "seed": 97531},
    )

    # 合法 P 更新绝不因诊断异常被标为 rejected_numerical。
    assert summary["status"] in {"accepted_improved", "accepted_reverted"}
    natural = summary["natural_validation"]
    assert natural["diagnostic_status"] == "failed_diagnostic"
    assert natural["diagnostic_failure_kind"] == "ValueError"
    assert "injected diagnostic failure" in str(natural["diagnostic_failure_detail"])
    # 失败绝不写成改善或“无恶化”。
    assert natural["natural_worsened"] == NE
    assert natural["natural_frozen_improved"] == NE
    assert natural["natural_fresh_greedy_improved"] == NE
    assert natural["natural_full_bellman_improvement"] == NE
    assert natural["natural_validation_regression_observed"] == NE
    # Model / Optimizer / RNG 不被诊断异常改变。
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    optimizer_state_after = {
        k: v for k, v in episode.optimizers["policy_value"].state_dict().items()
        if k != "state"
    }
    assert optimizer_state_before == optimizer_state_after


def test_e2_prepare_failure_isolated(monkeypatch):
    episode = _episode()
    model = episode.models["policy_value"]
    _set_head_bias(model, 2.0)
    train = _batch(episode.device, n_rows=12, eta1_rows=2, seed=53)
    validation = _batch(episode.device, n_rows=10, eta1_rows=1, seed=54)
    _stub_noop_optimizers(monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("injected prepare failure")

    monkeypatch.setattr(solver, "natural_diag_prepare", _boom)
    summary = _run_iteration(
        episode, model, [train], [validation],
        natural_validation_batches=[validation],
        natural_diag_mode="independent",
        natural_diag_config={"max_parents": 8, "chunk_size": 4, "seed": 97531},
    )
    assert summary["status"] in {"accepted_improved", "accepted_reverted"}
    natural = summary["natural_validation"]
    assert natural["diagnostic_status"] == "failed_diagnostic"
    assert natural["diagnostic_failure_kind"] == "RuntimeError"
    assert natural["natural_full_bellman_improvement"] == NE


# --- Test F：三种 Validation 分布 --------------------------------------------

def _rows(n, p0, pi, fg0, fg1, eta, source, finite=True):
    return {
        "p0": torch.zeros(n),
        "pi": torch.zeros(n),
        "p0_norm": torch.full((n,), float(p0)),
        "pi_norm": torch.full((n,), float(pi)),
        "fg_p0_norm": torch.full((n,), float(fg0)),
        "fg_pi_norm": torch.full((n,), float(fg1)),
        "eta": torch.tensor([float(e) for e in eta]),
        "source": torch.tensor([int(s) for s in source], dtype=torch.long),
        "row_finite": torch.full((n,), bool(finite), dtype=torch.bool),
    }


def _block(frozen, fresh, **kwargs):
    params = dict(
        mode="independent",
        natural_validation_reused=False,
        calibers_identical=False,
        sampling={},
        parity=None,
    )
    params.update(kwargs)
    return solver._assemble_natural_block(
        frozen_init=frozen,
        frozen_final=frozen,
        fresh_init=fresh,
        fresh_final=fresh,
        **params,
    )


def test_f_three_validation_views_by_source():
    # 4 个 Simulated(source0) + 2 个 Coverage(source1)；终值优于初值。
    eta = [0, 1, 0, 1, 0, 1]
    source = [0, 0, 0, 0, 1, 1]
    frozen_init = _rows(6, 2.0, 2.0, 2.0, 2.0, eta, source)
    frozen_final = _rows(6, 1.0, 1.0, 1.0, 1.0, eta, source)
    fresh_init = _rows(6, 2.0, 2.0, 2.0, 2.0, eta, source)
    fresh_final = _rows(6, 1.0, 1.0, 1.0, 1.0, eta, source)
    block = solver._assemble_natural_block(
        mode="independent",
        natural_validation_reused=False,
        calibers_identical=False,
        sampling={"selected_parents": 6},
        parity=None,
        frozen_init=frozen_init,
        frozen_final=frozen_final,
        fresh_init=fresh_init,
        fresh_final=fresh_final,
    )
    views = block["views"]
    mixture = views["original_mixture"]
    sim = views["simulated_only"]
    assert mixture["n_parents"] == 6
    assert mixture["eta0_count"] == 3 and mixture["eta1_count"] == 3
    assert mixture["eta1_share"] == 0.5
    assert mixture["evaluable"] is True
    assert sim["n_parents"] == 4
    assert sim["eta0_count"] == 2 and sim["eta1_count"] == 2
    assert sim["source"] == "simulated_only_source0"
    assert mixture["p0_frozen_mae"] == 1.0
    assert mixture["p0_fresh_greedy_mae"] == 1.0
    assert block["natural_frozen_improved"] is True
    assert block["natural_fresh_greedy_improved"] is True
    assert block["natural_full_bellman_improvement"] is True
    assert block["diagnostic_source_counts"]["source0"] == 4
    assert block["diagnostic_source_counts"]["source1"] == 2


def test_f2_empty_simulated_and_nonfinite_and_unpairable():
    # Source0 为空：Simulated-only 必须 not_evaluable，不得补样本。
    rows = _rows(3, 1.0, 1.0, 1.0, 1.0, [0, 0, 1], [1, 1, 1])
    block = _block(rows, rows)
    sim = block["views"]["simulated_only"]
    assert sim["status"] == NE
    assert sim["evaluable"] is False
    assert "p0_frozen_mae" not in sim

    # η=1 为空：如实报 0，不得虚构 η=1 统计。
    rows_eta = _rows(2, 1.0, 1.0, 1.0, 1.0, [0, 0], [0, 1])
    block_eta = _block(rows_eta, rows_eta)
    assert block_eta["eta1_count"] == 0
    assert block_eta["eta1_share"] == 0.0

    # 某 Source 的残差存在 NaN（行级 nonfinite）：该口径不可评估，改善判断
    # not_evaluable，绝不把失败写成零误差或改善。
    rows_nan = _rows(
        4, float("nan"), 1.0, 1.0, 1.0, [0, 1, 0, 1], [0, 0, 1, 1], finite=False
    )
    block_nan = _block(rows_nan, rows_nan)
    views_nan = block_nan["views"]
    assert views_nan["simulated_only"]["evaluable"] is False
    assert views_nan["simulated_only"]["nonfinite_ratio"] > 0.0
    assert block_nan["natural_frozen_improved"] == NE
    assert block_nan["natural_full_bellman_improvement"] == NE

    # 部分 Parent/Child 无法配对：如实丢弃并计数，绝不凭空补样本。
    episode = _episode()
    good = _batch(episode.device, n_rows=6, eta1_rows=1, seed=61)
    bad = _batch(episode.device, n_rows=6, eta1_rows=1, seed=62)
    bad["children"] = [bad["children"][0][:3], bad["children"][1][:3]]
    subset, sampling = solver._select_diagnostic_batches(
        [good, bad], max_parents=100, seed=1, episode_id=0,
    )
    assert sampling["dropped_unpairable_batches"] == 1
    assert sampling["requested_parents"] == 6
    assert len(subset) == 1
    assert int(subset[0]["parent"].shape[0]) == 6


# --- Test G：Natural Fresh Greedy 改善判断 -----------------------------------

def test_g_fresh_greedy_improvement_flags():
    def _pair(frozen_init_v, frozen_final_v, fresh_init_v, fresh_final_v):
        eta = [0, 1, 0, 1]
        source = [0, 1, 0, 1]
        frozen_init = _rows(4, frozen_init_v, frozen_init_v, 0, 0, eta, source)
        frozen_final = _rows(4, frozen_final_v, frozen_final_v, 0, 0, eta, source)
        fresh_init = _rows(4, 0, 0, fresh_init_v, fresh_init_v, eta, source)
        fresh_final = _rows(4, 0, 0, fresh_final_v, fresh_final_v, eta, source)
        return solver._diag_improvement_flags(
            frozen_init, frozen_final, fresh_init, fresh_final
        )

    # Frozen 改善 + Fresh Greedy 改善 → full improvement True。
    flags = _pair(2.0, 1.0, 2.0, 1.0)
    assert flags["natural_frozen_improved"] is True
    assert flags["natural_fresh_greedy_improved"] is True
    assert flags["natural_full_bellman_improvement"] is True

    # Frozen 改善 + Fresh Greedy 恶化 → 不得宣称完整 Bellman 改善。
    flags = _pair(2.0, 1.0, 1.0, 2.0)
    assert flags["natural_frozen_improved"] is True
    assert flags["natural_fresh_greedy_worsened"] is True
    assert flags["natural_full_bellman_improvement"] is False

    # Frozen 恶化 + Fresh Greedy 改善。
    flags = _pair(1.0, 2.0, 2.0, 1.0)
    assert flags["natural_frozen_worsened"] is True
    assert flags["natural_fresh_greedy_improved"] is True
    assert flags["natural_full_bellman_improvement"] is False

    # 诊断不可评估（非有限行）→ 三态 not_evaluable，绝不写 False/改善。
    eta = [0, 1, 0, 1]
    source = [0, 1, 0, 1]
    frozen_init = _rows(4, float("nan"), 1, 1, 1, eta, source, finite=False)
    frozen_final = _rows(4, 1, 1, 1, 1, eta, source)
    fresh_init = _rows(4, 1, 1, 1, 1, eta, source)
    fresh_final = _rows(4, 1, 1, 1, 1, eta, source)
    flags = solver._diag_improvement_flags(
        frozen_init, frozen_final, fresh_init, fresh_final
    )
    for key, value in flags.items():
        assert value == NE, key


def test_g2_revert_reports_not_improved_not_worsened(monkeypatch):
    # P Iteration Accepted Reverted：终值 == 初值 → 不改善也不恶化。
    episode = _episode()
    model = episode.models["policy_value"]
    _set_head_bias(model, 2.0)
    train = _batch(episode.device, n_rows=12, eta1_rows=2, seed=71)
    validation = _batch(episode.device, n_rows=10, eta1_rows=1, seed=72)
    _stub_noop_optimizers(monkeypatch)

    summary = _run_iteration(
        episode, model, [train], [validation],
        natural_validation_batches=[validation],
        natural_diag_mode="independent",
        natural_diag_config={"max_parents": 8, "chunk_size": 4, "seed": 97531},
    )
    assert summary["status"] == "accepted_reverted"
    natural = summary["natural_validation"]
    assert natural["diagnostic_status"] == "completed"
    assert natural["natural_frozen_improved"] is False
    assert natural["natural_frozen_worsened"] is False
    assert natural["natural_fresh_greedy_improved"] is False
    assert natural["natural_fresh_greedy_worsened"] is False
    assert natural["natural_full_bellman_improvement"] is False
    assert natural["natural_validation_regression_observed"] is False
    # Fresh Greedy Initial/Final 双时点可比较。
    assert natural["fresh_greedy_initial"] is not None
    assert natural["fresh_greedy_final"] is not None


# --- Test H：Old / Hybrid P 求解器分流 --------------------------------------

def test_h_old_p_and_hybrid_p_dispatch_are_isolated(monkeypatch):
    episode = _episode()
    batch = _batch(episode.device, n_rows=8, eta1_rows=1, seed=81)
    recorded = {"old": 0, "hybrid": 0}

    def _old(*args, **kwargs):
        recorded["old"] += 1
        return {"status": "accepted", "optimizer_steps": 1}

    def _hybrid(*args, **kwargs):
        recorded["hybrid"] += 1
        return {"status": "accepted", "optimizer_steps": 1}

    monkeypatch.setattr(episode, "_run_policy_value_evaluation_stage", _old)
    monkeypatch.setattr(episode, "_run_frozen_policy_p_stage", _hybrid)

    # Old P：fixed_target_cache 只走旧 Fixed Target Cache 路径。
    episode.hyperparams.p_solver = "fixed_target_cache"
    episode._run_p_stage_for_inner(
        [batch], [batch], episode.models["policy_value"], 1,
        inner_index=1, inner_cycles=1,
        natural_diag_request={"mode": "independent", "batches": [batch]},
    )
    assert recorded == {"old": 1, "hybrid": 0}

    # Hybrid P：frozen_policy_dynamic_residual 只走 dynamic residual 路径。
    episode.hyperparams.p_solver = "frozen_policy_dynamic_residual"
    episode._run_p_stage_for_inner(
        [batch], [batch], episode.models["policy_value"], 1,
        inner_index=1, inner_cycles=1,
        natural_diag_request={"mode": "independent", "batches": [batch]},
    )
    assert recorded == {"old": 1, "hybrid": 1}
