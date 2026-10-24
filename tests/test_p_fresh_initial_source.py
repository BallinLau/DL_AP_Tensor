"""PV_FRESH_INITIAL_SOURCE 执行层性能开关回归测试（fresh 初评口径来源）。

build_problem 与 _fresh_greedy_rows 在同一 model 快照上解同一贪心映射；
reuse_build 从已构建的冻结问题派生 fresh 初评残差（不再二次调用
teacher.compute_value_target），不改变训练 / Acceptance / 回滚语义。默认
resolve 保持现状逐位一致。

Test 1：reuse_build 派生的 fresh 初评行与 resolve 真实重解逐行一致（1e-6 内），
        行级字段（eta/source/row_finite）完全兼容 Natural 分组。
Test 2：reuse_build 全路径：fresh 初评统计与 resolve 一致（1e-6 内），初评不再
        调用 _fresh_greedy_rows，metadata 标注 fresh_initial_source=reuse_build。
Test 3：非法取值 fail fast（ValueError，消息清晰）。
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
    # Hybrid P solver：微小预算保持 CPU 测试快速。
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
    with torch.no_grad():
        for head in (model.v0_head, model.vi_head):
            linears = [m for m in head.network.modules() if isinstance(m, nn.Linear)]
            linears[-1].bias.fill_(float(value))


def _batch(device: torch.device, n_rows: int = 6, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    parent_state = torch.zeros(n_rows, 7, device=device)
    parent_state[:, 0] = torch.rand(n_rows, generator=g) * 0.5
    parent_state[:, 1] = torch.randn(n_rows, generator=g) * 0.3
    parent_state[:, 2] = 0.0
    parent_state[: max(n_rows // 2, 1), 2] = 1.0
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


def _run_iteration(episode: Episode, model, train, validation, **kwargs):
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


def _count_calls(monkeypatch, name: str):
    calls = {"n": 0}
    original = getattr(solver, name)

    def _wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        calls["n"] += 1
        return result

    monkeypatch.setattr(solver, name, _wrapped)
    return calls


# --- Test 1: reuse_build 行级与 resolve 一致 ---------------------------------

def test_fresh_initial_rows_reuse_build_match_resolve():
    episode = _episode()
    model = episode.models["policy_value"]
    _set_head_bias(model, 2.0)
    teacher = _teacher(episode)
    batches = [
        _batch(episode.device, n_rows=6, seed=1),
        _batch(episode.device, n_rows=5, seed=2),
    ]
    resolve_rows = solver._fresh_greedy_rows(
        episode, model, teacher, batches, q_model=model
    )
    problem, _parity = solver.build_problem(
        episode, model, teacher, batches, q_model=model
    )
    reuse_rows = solver._fresh_rows_from_frozen_problem(model, problem)

    assert set(reuse_rows) == set(resolve_rows)
    for key in ("fg_p0_norm", "fg_pi_norm"):
        assert torch.allclose(
            resolve_rows[key], reuse_rows[key], atol=1e-6, rtol=0.0
        ), key
    # Natural 分组依赖的行级字段语义完全兼容。
    for key in ("eta", "source", "row_finite"):
        assert torch.equal(resolve_rows[key], reuse_rows[key]), key


# --- Test 2: reuse_build 全路径 ----------------------------------------------

def test_fresh_initial_source_reuse_build_reuses_built_problem(monkeypatch):
    stats = {}
    for source in ("resolve", "reuse_build"):
        episode = _episode()
        model = episode.models["policy_value"]
        _set_head_bias(model, 2.0)
        episode.hyperparams.pv_fresh_initial_source = source
        batch = _batch(episode.device, n_rows=6, seed=1)
        fresh_calls = _count_calls(monkeypatch, "_fresh_greedy_rows")
        summary = _run_iteration(episode, model, [batch], [batch])
        stats[source] = (summary["fresh_greedy_initial"], fresh_calls["n"], summary)

    resolve_stats, resolve_calls, resolve_summary = stats["resolve"]
    reuse_stats, reuse_calls, reuse_summary = stats["reuse_build"]
    # resolve：初评 + 终评两次真实重解；reuse_build：仅终评，初评从冻结问题派生。
    assert resolve_calls == 2
    assert reuse_calls == 1
    assert "fresh_initial_source" not in resolve_stats
    assert reuse_stats["fresh_initial_source"] == "reuse_build"
    for key in (
        "fresh_greedy_p0_norm_mae",
        "fresh_greedy_pi_norm_mae",
        "fresh_greedy_p0_norm_p90",
        "fresh_greedy_pi_norm_p90",
    ):
        assert abs(float(resolve_stats[key]) - float(reuse_stats[key])) <= 1e-6, key
    # Natural Validation 行级分组在 reuse_build 下照常组装。
    assert reuse_summary["natural_validation"] is not None
    assert reuse_summary["natural_validation"]["fresh_greedy_initial"] is not None
    assert reuse_summary["status"] == resolve_summary["status"]


# --- Test 3: 非法取值 fail fast ----------------------------------------------

def test_illegal_perf_switch_values_fail_fast():
    episode = _episode()
    model = episode.models["policy_value"]
    _set_head_bias(model, 2.0)
    episode.hyperparams.pv_fresh_initial_source = "lazy"
    batch = _batch(episode.device, n_rows=4, seed=1)
    with pytest.raises(ValueError, match="pv_fresh_initial_source"):
        _run_iteration(episode, model, [batch], [batch])
